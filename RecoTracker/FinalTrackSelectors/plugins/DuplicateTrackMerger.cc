/** \class DuplicateTrackMerger
 * 
 * selects pairs of tracks that should be single tracks
 *
 * \author Matthew Walker
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "TrackMerger.h"
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

using namespace reco;

    class dso_hidden DuplicateTrackMerger final : public edm::stream::EDProducer<> {
       public:
         /// constructor
         explicit DuplicateTrackMerger(const edm::ParameterSet& iPara);
	 /// destructor
	 virtual ~DuplicateTrackMerger();

	 typedef std::vector<std::pair<TrackCandidate,std::pair<reco::TrackRef,reco::TrackRef> > > CandidateToDuplicate;

       protected:
	 /// produce one event
	 void produce( edm::Event &, const edm::EventSetup &) override;

       private:
	 /// MVA discriminator
	 const GBRForest* forest_;

	 /// MVA input variables
	 float tmva_ddsz_;
	 float tmva_ddxy_;
	 float tmva_dphi_;
	 float tmva_dlambda_;
	 float tmva_dqoverp_;
	 float tmva_d3dr_;
	 float tmva_d3dz_;
	 float tmva_outer_nMissingInner_;
	 float tmva_inner_nMissingOuter_;

	 float* gbrVals_;

	 /// track input collection
	 edm::EDGetTokenT<reco::TrackCollection> trackSource_;
	 /// MVA weights file
	 std::string dbFileName_;
	 bool useForestFromDB_;
	 std::string forestLabel_;
	 /// minDeltaR3d cut value
	 double minDeltaR3d_;
	 /// minBDTG cut value
	 double minBDTG_;
	 ///min pT cut value
	 double minpT_;
	 ///min p cut value
	 double minP_;
	 ///max distance between two tracks at closest approach
	 double maxDCA_;
	 ///max difference in phi between two tracks
	 double maxDPhi_;
	 ///max difference in Lambda between two tracks
	 double maxDLambda_;
	 ///max difference in transverse impact parameter between two tracks
	 double maxDdxy_;
	 ///max difference in longitudinal impact parameter between two tracks
	 double maxDdsz_;
	 ///max difference in q/p between two tracks
	 double maxDQoP_;

	 edm::ESHandle<MagneticField> magfield_;

	 ///Merger
	 TrackMerger merger_;
     };

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "TFile.h"

DuplicateTrackMerger::DuplicateTrackMerger(const edm::ParameterSet& iPara) : forest_(nullptr), gbrVals_(nullptr), merger_(iPara)
{
  forestLabel_ = "MVADuplicate";
  useForestFromDB_ = true;

  minDeltaR3d_ = -4.0;
  minBDTG_ = -0.96;
  minpT_ = 0.2;
  minP_ = 0.4;
  maxDCA_ = 30.0;
  maxDPhi_ = 0.30;
  maxDLambda_ = 0.30;
  maxDdsz_ = 10.0;
  maxDdxy_ = 10.0;
  maxDQoP_ = 0.25;
  if(iPara.exists("minpT"))minpT_ = iPara.getParameter<double>("minpT");
  if(iPara.exists("minP"))minP_ = iPara.getParameter<double>("minP");
  if(iPara.exists("maxDCA"))maxDCA_ = iPara.getParameter<double>("maxDCA");
  if(iPara.exists("maxDPhi"))maxDPhi_ = iPara.getParameter<double>("maxDPhi");
  if(iPara.exists("maxDLambda"))maxDLambda_ = iPara.getParameter<double>("maxDLambda");
  if(iPara.exists("maxDdsz"))maxDdsz_ = iPara.getParameter<double>("maxDdsz");
  if(iPara.exists("maxDdxy"))maxDdxy_ = iPara.getParameter<double>("maxDdxy");
  if(iPara.exists("maxDQoP"))maxDQoP_ = iPara.getParameter<double>("maxDQoP");
  if(iPara.exists("source"))trackSource_ = consumes<reco::TrackCollection>(iPara.getParameter<edm::InputTag>("source"));
  if(iPara.exists("minDeltaR3d"))minDeltaR3d_ = iPara.getParameter<double>("minDeltaR3d");
  if(iPara.exists("minBDTG"))minBDTG_ = iPara.getParameter<double>("minBDTG");

  produces<std::vector<TrackCandidate> >("candidates");
  produces<CandidateToDuplicate>("candidateMap");

  gbrVals_ = new float[9];
  dbFileName_ = "";
  if(iPara.exists("GBRForestLabel"))forestLabel_ = iPara.getParameter<std::string>("GBRForestLabel");
  if(iPara.exists("GBRForestFileName")){
    dbFileName_ = iPara.getParameter<std::string>("GBRForestFileName");
    useForestFromDB_ = false;
  }

  /*
  tmvaReader_ = new TMVA::Reader("!Color:Silent");
  tmvaReader_->AddVariable("ddsz",&tmva_ddsz_);
  tmvaReader_->AddVariable("ddxy",&tmva_ddxy_);
  tmvaReader_->AddVariable("dphi",&tmva_dphi_);
  tmvaReader_->AddVariable("dlambda",&tmva_dlambda_);
  tmvaReader_->AddVariable("dqoverp",&tmva_dqoverp_);
  tmvaReader_->AddVariable("delta3d_r",&tmva_d3dr_);
  tmvaReader_->AddVariable("delta3d_z",&tmva_d3dz_);
  tmvaReader_->AddVariable("outer_nMissingInner",&tmva_outer_nMissingInner_);
  tmvaReader_->AddVariable("inner_nMissingOuter",&tmva_inner_nMissingOuter_);
  tmvaReader_->BookMVA("BDTG",mvaFilePath);
  */

}

DuplicateTrackMerger::~DuplicateTrackMerger()
{

  if(gbrVals_)delete [] gbrVals_;
  if(!useForestFromDB_ && forest_) delete forest_;
  /* no op */

}

void DuplicateTrackMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  merger_.init(iSetup);

  if(!forest_){
    if(useForestFromDB_){
      edm::ESHandle<GBRForest> forestHandle;
      iSetup.get<GBRWrapperRcd>().get(forestLabel_,forestHandle);
      forest_ = forestHandle.product();
    }else{
      TFile gbrfile(dbFileName_.c_str());
      forest_ = dynamic_cast<const GBRForest*>(gbrfile.Get(forestLabel_.c_str()));
    }
  }

  //edm::Handle<edm::View<reco::Track> >handle;
  edm::Handle<reco::TrackCollection >handle;
  iEvent.getByToken(trackSource_,handle);
  reco::TrackRefProd refTrks(handle);

  iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
  TwoTrackMinimumDistance ttmd;
  TSCPBuilderNoMaterial tscpBuilder;
  std::auto_ptr<std::vector<TrackCandidate> > out_duplicateCandidates(new std::vector<TrackCandidate>());

  std::auto_ptr<CandidateToDuplicate> out_candidateMap(new CandidateToDuplicate());

  for(int i = 0; i < (int)handle->size(); i++){
    const reco::Track *rt1 = &(handle->at(i));
    if(rt1->innerMomentum().Rho() < minpT_)continue;
    if(rt1->innerMomentum().R() < minP_)continue;
    for(int j = i+1; j < (int)handle->size();j++){
      const reco::Track *rt2 = &(handle->at(j));
      if(rt2->innerMomentum().Rho() < minpT_)continue;
      if(rt2->innerMomentum().R() < minP_)continue;
      if(rt1->charge() != rt2->charge())continue;
      const reco::Track* t1,*t2;
      if(rt1->outerPosition().Rho() < rt2->outerPosition().Rho()){
	t1 = rt1;
	t2 = rt2;
      }else{
	t1 = rt2;
	t2 = rt1;
      }
      double deltaR3d = sqrt(pow(t1->outerPosition().x()-t2->innerPosition().x(),2) + pow(t1->outerPosition().y()-t2->innerPosition().y(),2) + pow(t1->outerPosition().z()-t2->innerPosition().z(),2));

      if(t1->outerPosition().Rho() > t2->innerPosition().Rho())deltaR3d *= -1.0;
      if(deltaR3d < minDeltaR3d_)continue;
      
      FreeTrajectoryState fts1 = trajectoryStateTransform::outerFreeState(*t1, &*magfield_,false);
      FreeTrajectoryState fts2 = trajectoryStateTransform::innerFreeState(*t2, &*magfield_,false);
      GlobalPoint avgPoint((t1->outerPosition().x()+t2->innerPosition().x())*0.5,(t1->outerPosition().y()+t2->innerPosition().y())*0.5,(t1->outerPosition().z()+t2->innerPosition().z())*0.5);
      TrajectoryStateClosestToPoint TSCP1 = tscpBuilder(fts1, avgPoint);
      TrajectoryStateClosestToPoint TSCP2 = tscpBuilder(fts2, avgPoint);
      if(!TSCP1.isValid())continue;
      if(!TSCP2.isValid())continue;

      const FreeTrajectoryState ftsn1 = TSCP1.theState();
      const FreeTrajectoryState ftsn2 = TSCP2.theState();
 
      if ( (ftsn2.position()-ftsn1.position()).mag() > maxDCA_ ) continue;

      double qoverp1 = ftsn1.signedInverseMomentum();
      double qoverp2 = ftsn2.signedInverseMomentum();
      tmva_dqoverp_ = qoverp1-qoverp2;
      if ( fabs(tmva_dqoverp_) > maxDQoP_ ) continue;

      double lambda1 =  M_PI/2 - ftsn1.momentum().theta();
      double lambda2 =  M_PI/2 - ftsn2.momentum().theta();
      tmva_dlambda_ = lambda1-lambda2;
      if ( fabs(tmva_dlambda_) > maxDLambda_ ) continue;

      double phi1 = ftsn1.momentum().phi();
      double phi2 = ftsn2.momentum().phi();
      tmva_dphi_ = phi1-phi2;
      if(fabs(tmva_dphi_) > M_PI) tmva_dphi_ = 2*M_PI - fabs(tmva_dphi_);
      if ( fabs(tmva_dphi_) > maxDPhi_ ) continue;

      double dxy1 = (-ftsn1.position().x() * ftsn1.momentum().y() + ftsn1.position().y() * ftsn1.momentum().x())/TSCP1.pt();
      double dxy2 = (-ftsn2.position().x() * ftsn2.momentum().y() + ftsn2.position().y() * ftsn2.momentum().x())/TSCP2.pt();
      tmva_ddxy_ = dxy1-dxy2;
      if ( fabs(tmva_ddxy_) > maxDdxy_ ) continue;

      double dsz1 = ftsn1.position().z() * TSCP1.pt() / TSCP1.momentum().mag() - (ftsn1.position().x() * ftsn1.momentum().y() + ftsn1.position().y() * ftsn1.momentum().x())/TSCP1.pt() * ftsn1.momentum().z()/ftsn1.momentum().mag();
      double dsz2 = ftsn2.position().z() * TSCP2.pt() / TSCP2.momentum().mag() - (ftsn2.position().x() * ftsn2.momentum().y() + ftsn2.position().y() * ftsn2.momentum().x())/TSCP2.pt() * ftsn2.momentum().z()/ftsn2.momentum().mag();
      tmva_ddsz_ = dsz1-dsz2;
      if ( fabs(tmva_ddsz_) > maxDdsz_ ) continue;

      tmva_d3dr_ = avgPoint.perp();
      tmva_d3dz_ = avgPoint.z();
      tmva_outer_nMissingInner_ = t2->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      tmva_inner_nMissingOuter_ = t1->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
      

      gbrVals_[0] = tmva_ddsz_;
      gbrVals_[1] = tmva_ddxy_;
      gbrVals_[2] = tmva_dphi_;
      gbrVals_[3] = tmva_dlambda_;
      gbrVals_[4] = tmva_dqoverp_;
      gbrVals_[5] = tmva_d3dr_;
      gbrVals_[6] = tmva_d3dz_;
      gbrVals_[7] = tmva_outer_nMissingInner_;
      gbrVals_[8] = tmva_inner_nMissingOuter_;


      double mvaBDTG = forest_->GetClassifier(gbrVals_);
      if(mvaBDTG < minBDTG_)continue;
      
      
      TrackCandidate mergedTrack = merger_.merge(*t1,*t2);
      out_duplicateCandidates->push_back(mergedTrack);
      std::pair<TrackRef,TrackRef> trackPair(TrackRef(refTrks,i),TrackRef(refTrks,j));
      std::pair<TrackCandidate, std::pair<TrackRef,TrackRef> > cp(mergedTrack,trackPair);
      out_candidateMap->push_back(cp);
    }
  }
  iEvent.put(out_duplicateCandidates,"candidates");
  iEvent.put(out_candidateMap,"candidateMap");

}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DuplicateTrackMerger);

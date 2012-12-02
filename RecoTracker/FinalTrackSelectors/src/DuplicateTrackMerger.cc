#include "RecoTracker/FinalTrackSelectors/interface/DuplicateTrackMerger.h"

#include "FWCore/Framework/interface/Event.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

using reco::modules::DuplicateTrackMerger;

DuplicateTrackMerger::DuplicateTrackMerger(const edm::ParameterSet& iPara) : merger_(iPara)
{
  weightFileName_ = "RecoTracker/FinalTrackSelectors/data/DuplicateWeights.xml";
  minDeltaR3d_ = -4.0;
  minBDTG_ = -0.96;
  minpT_ = 0.2;
  minP_ = 0.4;
  maxDCA_ = 50.0;
  maxDPhi_ = 0.35;
  maxDLambda_ = 0.35;
  maxDdsz_ = 20.0;
  maxDdxy_ = 20.0;
  maxDQoP_ = 0.25;
  if(iPara.exists("minpT"))minpT_ = iPara.getParameter<double>("minpT");
  if(iPara.exists("minP"))minP_ = iPara.getParameter<double>("minP");
  if(iPara.exists("maxDCA"))maxDCA_ = iPara.getParameter<double>("maxDCA");
  if(iPara.exists("maxDPhi"))maxDPhi_ = iPara.getParameter<double>("maxDPhi");
  if(iPara.exists("maxDLambda"))maxDLambda_ = iPara.getParameter<double>("maxDLambda");
  if(iPara.exists("maxDdsz"))maxDdsz_ = iPara.getParameter<double>("maxDdsz");
  if(iPara.exists("maxDdxy"))maxDdxy_ = iPara.getParameter<double>("maxDdxy");
  if(iPara.exists("maxDQoP"))maxDQoP_ = iPara.getParameter<double>("maxDQoP");
  if(iPara.exists("source"))trackSource_ = iPara.getParameter<edm::InputTag>("source");
  if(iPara.exists("minDeltaR3d"))minDeltaR3d_ = iPara.getParameter<double>("minDeltaR3d");
  if(iPara.exists("minBDTG"))minBDTG_ = iPara.getParameter<double>("minBDTG");
  if(iPara.exists("weightsFile"))weightFileName_ = iPara.getParameter<std::string>("weightsFile");

  produces<std::vector<TrackCandidate> >("candidates");
  produces<CandidateToDuplicate>("candidateMap");

  std::string mvaFilePath = edm::FileInPath ( weightFileName_.c_str() ).fullPath();

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


}

DuplicateTrackMerger::~DuplicateTrackMerger()
{

  /* no op */

}

void DuplicateTrackMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  merger_.init(iSetup);

  edm::Handle<edm::View<reco::Track> >handle;
  iEvent.getByLabel(trackSource_,handle);

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
      
      FreeTrajectoryState fts1 = trajectoryStateTransform::initialFreeState(*t1, &*magfield_);
      FreeTrajectoryState fts2 = trajectoryStateTransform::initialFreeState(*t2, &*magfield_);
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
      tmva_outer_nMissingInner_ = t2->trackerExpectedHitsInner().numberOfLostHits();
      tmva_inner_nMissingOuter_ = t1->trackerExpectedHitsOuter().numberOfLostHits();
      
      double mvaBDTG = tmvaReader_->EvaluateMVA("BDTG");
      if(mvaBDTG < minBDTG_)continue;
      
      
      TrackCandidate mergedTrack = merger_.merge(*t1,*t2);
      out_duplicateCandidates->push_back(mergedTrack);
      std::pair<Track,Track> trackPair(*t1,*t2);
      std::pair<TrackCandidate, std::pair<Track,Track> > cp(mergedTrack,trackPair);
      out_candidateMap->push_back(cp);
    }
  }
  iEvent.put(out_duplicateCandidates,"candidates");
  iEvent.put(out_candidateMap,"candidateMap");

}

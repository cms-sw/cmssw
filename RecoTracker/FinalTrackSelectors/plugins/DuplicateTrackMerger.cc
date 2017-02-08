/** \class DuplicateTrackMerger
 * 
 * selects pairs of tracks that should be single tracks
 *
 * \author Matthew Walker
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"


#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"

#include "TrackMerger.h"

#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <atomic>

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

using namespace reco;
namespace {
  class  DuplicateTrackMerger final : public edm::stream::EDProducer<> {
  public:
    /// constructor
    explicit DuplicateTrackMerger(const edm::ParameterSet& iPara);
    /// destructor
    virtual ~DuplicateTrackMerger();
    
    using CandidateToDuplicate = std::vector<std::pair<int, int>>;
	 
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

	 
    private:
      /// produce one event
      void produce( edm::Event &, const edm::EventSetup &) override;
      
    private:
      /// MVA discriminator
      const GBRForest* forest_;
      
           
      /// MVA weights file
      std::string dbFileName_;
      bool useForestFromDB_;
      std::string forestLabel_;
      
      
      /// track input collection
      edm::EDGetTokenT<reco::TrackCollection> trackSource_;
      /// minDeltaR3d cut value
      double minDeltaR3d2_;
      /// minBDTG cut value
      double minBDTG_;
      ///min pT cut value
      double minpT2_;
      ///min p cut value
      double minP_;
      ///max distance between two tracks at closest approach
      float maxDCA2_;
      ///max difference in phi between two tracks
      float maxDPhi_;
      ///max difference in Lambda between two tracks
      float maxDLambda_;
      ///max difference in transverse impact parameter between two tracks
      float maxDdxy_;
      ///max difference in longitudinal impact parameter between two tracks
      float maxDdsz_;
      ///max difference in q/p between two tracks
      float maxDQoP_;
      
      edm::ESHandle<MagneticField> magfield_;
      
      ///Merger
      TrackMerger merger_;
    };
  }
    
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

namespace {


void
DuplicateTrackMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
     edm::ParameterSetDescription desc;
     desc.add<edm::InputTag>("source",edm::InputTag());
     desc.add<double>("minDeltaR3d",-4.0);
     desc.add<double>("minBDTG",-0.1);
     desc.add<double>("minpT",0.2);
     desc.add<double>("minP",0.4);
     desc.add<double>("maxDCA",30.0);
     desc.add<double>("maxDPhi",0.30);
     desc.add<double>("maxDLambda",0.30);
     desc.add<double>("maxDdsz",10.0);
     desc.add<double>("maxDdxy",10.0);
     desc.add<double>("maxDQoP",0.25);
     desc.add<std::string>("forestLabel","MVADuplicate");
     desc.add<std::string>("GBRForestFileName","");
     desc.add<bool>("useInnermostState",true);
     desc.add<std::string>("ttrhBuilderName","WithAngleAndTemplate");
     descriptions.add("DuplicateTrackMerger", desc);
}

  
DuplicateTrackMerger::DuplicateTrackMerger(const edm::ParameterSet& iPara) : forest_(nullptr), merger_(iPara)
{

  trackSource_ = consumes<reco::TrackCollection>(iPara.getParameter<edm::InputTag>("source"));
  minDeltaR3d2_ = iPara.getParameter<double>("minDeltaR3d"); minDeltaR3d2_*=std::abs(minDeltaR3d2_);
  minBDTG_ = iPara.getParameter<double>("minBDTG");
  minpT2_ = iPara.getParameter<double>("minpT"); minpT2_ *= minpT2_;
  minP_ = iPara.getParameter<double>("minP");
  maxDCA2_ = iPara.getParameter<double>("maxDCA"); maxDCA2_*=maxDCA2_;
  maxDPhi_ = iPara.getParameter<double>("maxDPhi");
  maxDLambda_ = iPara.getParameter<double>("maxDLambda");
  maxDdsz_ = iPara.getParameter<double>("maxDdsz");
  maxDdxy_ = iPara.getParameter<double>("maxDdxy");
  maxDQoP_ = iPara.getParameter<double>("maxDQoP");

  produces<std::vector<TrackCandidate> >("candidates");
  produces<CandidateToDuplicate>("candidateMap");

  forestLabel_ = iPara.getParameter<std::string>("forestLabel");

  dbFileName_ = iPara.getParameter<std::string>("GBRForestFileName");
  useForestFromDB_ = dbFileName_.empty();

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

  if(!useForestFromDB_) delete forest_;

}


#ifdef VI_STAT
  struct Stat {
    Stat() : maxCos(1.1), nCand(0),nLoop0(0) {}
    ~Stat() {
      std::cout << "Stats " << nCand << ' ' << nLoop0 << ' ' << maxCos << std::endl;
    }
    std::atomic<float> maxCos;
    std::atomic<int> nCand, nLoop0;
  };
  Stat stat;
#endif
  
template<typename T>
void update_maximum(std::atomic<T>& maximum_value, T const& value) noexcept
{
    T prev_value = maximum_value;
    while(prev_value < value &&
            !maximum_value.compare_exchange_weak(prev_value, value))
        ;
}

  template<typename T>
void update_minimum(std::atomic<T>& minimum_value, T const& value) noexcept
{
    T prev_value = minimum_value;
    while(prev_value > value &&
            !minimum_value.compare_exchange_weak(prev_value, value))
        ;
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
  auto const & tracks = *handle;
  
  iSetup.get<IdealMagneticFieldRecord>().get(magfield_);
  TwoTrackMinimumDistance ttmd;
  TSCPBuilderNoMaterial tscpBuilder;
  auto out_duplicateCandidates = std::make_unique<std::vector<TrackCandidate>>();

  auto out_candidateMap = std::make_unique<CandidateToDuplicate>();

  for(int i = 0; i < (int)tracks.size(); i++){
    const reco::Track *rt1 = &tracks[i];
    if(rt1->innerMomentum().perp2() < minpT2_)continue;
    // if(rt1->innerMomentum().R() < minP_)continue;
    for(int j = i+1; j < (int)tracks.size();j++){
      const reco::Track *rt2 = &tracks[j];
      if(rt1->charge() != rt2->charge())continue;
      auto cosT = (*rt1).momentum().unit().Dot((*rt2).momentum().unit());
      if (cosT<0.) continue;
      if(rt2->innerMomentum().perp2() < minpT2_)continue;
      // if(rt2->innerMomentum().R() < minP_)continue;
      const reco::Track* t1,*t2;
      if(rt1->outerPosition().perp2() < rt2->outerPosition().perp2()){
	t1 = rt1;
	t2 = rt2;
      }else{
	t1 = rt2;
	t2 = rt1;
      }
      auto deltaR3d2 = (t1->outerPosition() - t2->innerPosition()).mag2();
      
      if(t1->outerPosition().perp2() > t2->innerPosition().perp2()) deltaR3d2 *= -1.0;
      if(deltaR3d2 < minDeltaR3d2_)continue;
      
      FreeTrajectoryState fts1 = trajectoryStateTransform::outerFreeState(*t1, &*magfield_,false);
      FreeTrajectoryState fts2 = trajectoryStateTransform::innerFreeState(*t2, &*magfield_,false);
      GlobalPoint avgPoint((t1->outerPosition().x()+t2->innerPosition().x())*0.5,(t1->outerPosition().y()+t2->innerPosition().y())*0.5,(t1->outerPosition().z()+t2->innerPosition().z())*0.5);
      TrajectoryStateClosestToPoint TSCP1 = tscpBuilder(fts1, avgPoint);
      TrajectoryStateClosestToPoint TSCP2 = tscpBuilder(fts2, avgPoint);
      if(!TSCP1.isValid())continue;
      if(!TSCP2.isValid())continue;

      const FreeTrajectoryState ftsn1 = TSCP1.theState();
      const FreeTrajectoryState ftsn2 = TSCP2.theState();
 
      if ( (ftsn2.position()-ftsn1.position()).mag2() > maxDCA2_ ) continue;

      auto qoverp1 = ftsn1.signedInverseMomentum();
      auto qoverp2 = ftsn2.signedInverseMomentum();
      float tmva_dqoverp_ = qoverp1-qoverp2;
      if ( std::abs(tmva_dqoverp_) > maxDQoP_ ) continue;


      //auto pp = [&](TrajectoryStateClosestToPoint const & ts) { std::cout << ' ' << ts.perigeeParameters().vector()[0] << '/'  << ts.perigeeError().transverseCurvatureError();};
      //if(qoverp1*qoverp2 <0) { std::cout << "charge different " << qoverp1 <<',' << qoverp2; pp(TSCP1); pp(TSCP2); std::cout << std::endl;}
      
      auto lambda1 =  M_PI/2 - ftsn1.momentum().theta();
      auto lambda2 =  M_PI/2 - ftsn2.momentum().theta();
      float tmva_dlambda_ = lambda1-lambda2;
      if ( std::abs(tmva_dlambda_) > maxDLambda_ ) continue;

      auto phi1 = ftsn1.momentum().phi();
      auto phi2 = ftsn2.momentum().phi();
      float tmva_dphi_ = phi1-phi2;
      if(std::abs(tmva_dphi_) > float(M_PI)) tmva_dphi_ = 2.f*float(M_PI) - std::abs(tmva_dphi_);
      if (std::abs(tmva_dphi_) > maxDPhi_ ) continue;

      auto dxy1 = (-ftsn1.position().x() * ftsn1.momentum().y() + ftsn1.position().y() * ftsn1.momentum().x())/TSCP1.pt();
      auto dxy2 = (-ftsn2.position().x() * ftsn2.momentum().y() + ftsn2.position().y() * ftsn2.momentum().x())/TSCP2.pt();
      float tmva_ddxy_ = dxy1-dxy2;
      if ( std::abs(tmva_ddxy_) > maxDdxy_ ) continue;

      auto dsz1 = ftsn1.position().z() * TSCP1.pt() / TSCP1.momentum().mag()
	- (ftsn1.position().x() * ftsn1.momentum().y() + ftsn1.position().y() * ftsn1.momentum().x())/TSCP1.pt() * ftsn1.momentum().z()/ftsn1.momentum().mag();
      auto dsz2 = ftsn2.position().z() * TSCP2.pt() / TSCP2.momentum().mag()
	- (ftsn2.position().x() * ftsn2.momentum().y() + ftsn2.position().y() * ftsn2.momentum().x())/TSCP2.pt() * ftsn2.momentum().z()/ftsn2.momentum().mag();
      float tmva_ddsz_ = dsz1-dsz2;
      if ( std::abs(tmva_ddsz_) > maxDdsz_ ) continue;

      float tmva_d3dr_ = avgPoint.perp();
      float tmva_d3dz_ = avgPoint.z();
      float tmva_outer_nMissingInner_ = t2->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      float tmva_inner_nMissingOuter_ = t1->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_OUTER_HITS);
      
      float gbrVals_[9];
      gbrVals_[0] = tmva_ddsz_;
      gbrVals_[1] = tmva_ddxy_;
      gbrVals_[2] = tmva_dphi_;
      gbrVals_[3] = tmva_dlambda_;
      gbrVals_[4] = tmva_dqoverp_;
      gbrVals_[5] = tmva_d3dr_;
      gbrVals_[6] = tmva_d3dz_;
      gbrVals_[7] = tmva_outer_nMissingInner_;
      gbrVals_[8] = tmva_inner_nMissingOuter_;


      auto mvaBDTG = forest_->GetClassifier(gbrVals_);
      if(mvaBDTG < minBDTG_)continue;

      //  std::cout << "to merge " << mvaBDTG << ' ' << std::copysign(std::sqrt(std::abs(deltaR3d2)),deltaR3d2) << ' ' << tmva_dphi_ << ' ' << TSCP1.pt() <<'/'<<TSCP2.pt() << std::endl;
      
      out_duplicateCandidates->push_back(merger_.merge(*t1,*t2));
      out_candidateMap->emplace_back(i,j);

#ifdef VI_STAT
      ++stat.nCand;
      //    auto cosT = float((*t1).momentum().unit().Dot((*t2).momentum().unit()));
      if (cosT>0) update_minimum(stat.maxCos,float(cosT));
      else   ++stat.nLoop0;
#endif
      
    }
  }
  iEvent.put(std::move(out_duplicateCandidates),"candidates");
  iEvent.put(std::move(out_candidateMap),"candidateMap");

  
}


  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DuplicateTrackMerger);

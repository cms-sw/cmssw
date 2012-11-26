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
  minpT_ = 0.63;
  if(iPara.exists("minpT"))minpT_ = iPara.getParameter<double>("minpT");
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
    reco::Track rt1 = (handle->at(i));
    if(rt1.innerMomentum().Rho() < minpT_)continue;
    for(int j = i+1; j < (int)handle->size();j++){
      reco::Track rt2 = (handle->at(j));
      if(rt2.innerMomentum().Rho() < minpT_)continue;
      if(rt1.charge() != rt2.charge())continue;
      reco::Track t1,t2;
      if(rt1.outerPosition().Rho() < rt2.outerPosition().Rho()){
	t1 = rt1;
	t2 = rt2;
      }else{
	t1 = rt2;
	t2 = rt1;
      }
      double deltaR3d = sqrt(pow(t1.outerPosition().x()-t2.innerPosition().x(),2) + pow(t1.outerPosition().y()-t2.innerPosition().y(),2) + pow(t1.outerPosition().z()-t2.innerPosition().z(),2));

      if(t1.outerPosition().Rho() > t2.innerPosition().Rho())deltaR3d *= -1.0;
      if(deltaR3d < minDeltaR3d_)continue;
      
      FreeTrajectoryState fts1 = trajectoryStateTransform::initialFreeState(t1, &*magfield_);
      FreeTrajectoryState fts2 = trajectoryStateTransform::initialFreeState(t2, &*magfield_);
      GlobalPoint avgPoint((t1.outerPosition().x()+t2.innerPosition().x())*0.5,(t1.outerPosition().y()+t2.innerPosition().y())*0.5,(t1.outerPosition().z()+t2.innerPosition().z())*0.5);
      TrajectoryStateClosestToPoint TSCP1 = tscpBuilder(fts1, avgPoint);
      TrajectoryStateClosestToPoint TSCP2 = tscpBuilder(fts2, avgPoint);
      if(!TSCP1.isValid())continue;
      if(!TSCP2.isValid())continue;
              
      double qoverp1 = TSCP1.theState().signedInverseMomentum();
      double phi1 = TSCP1.theState().momentum().phi();
      double lambda1 =  M_PI/2 - TSCP1.theState().momentum().theta();
      double dxy1 = (-TSCP1.theState().position().x() * TSCP1.theState().momentum().y() + TSCP1.theState().position().y() * TSCP1.theState().momentum().x())/TSCP1.pt();
      double dsz1 = TSCP1.theState().position().z() * TSCP1.pt() / TSCP1.momentum().mag() - (TSCP1.theState().position().x() * TSCP1.theState().momentum().y() + TSCP1.theState().position().y() * TSCP1.theState().momentum().x())/TSCP1.pt() * TSCP1.theState().momentum().z()/TSCP1.theState().momentum().mag();
      
      double qoverp2 = TSCP2.theState().signedInverseMomentum();
      double phi2 = TSCP2.theState().momentum().phi();
      double lambda2 =  M_PI/2 - TSCP2.theState().momentum().theta();
      double dxy2 = (-TSCP2.theState().position().x() * TSCP2.theState().momentum().y() + TSCP2.theState().position().y() * TSCP2.theState().momentum().x())/TSCP2.pt();
      double dsz2 = TSCP2.theState().position().z() * TSCP2.pt() / TSCP2.momentum().mag() - (TSCP2.theState().position().x() * TSCP2.theState().momentum().y() + TSCP2.theState().position().y() * TSCP2.theState().momentum().x())/TSCP2.pt() * TSCP2.theState().momentum().z()/TSCP2.theState().momentum().mag();
      tmva_ddsz_ = dsz1-dsz2;
      tmva_ddxy_ = dxy1-dxy2;
      tmva_dphi_ = phi1-phi2;
      if(fabs(tmva_dphi_) > M_PI) tmva_dphi_ = 2*M_PI - fabs(tmva_dphi_);
      tmva_dlambda_ = lambda1-lambda2;
      tmva_dqoverp_ = qoverp1-qoverp2;
      tmva_d3dr_ = avgPoint.perp();
      tmva_d3dz_ = avgPoint.z();
      tmva_outer_nMissingInner_ = t2.trackerExpectedHitsInner().numberOfLostHits();
      tmva_inner_nMissingOuter_ = t1.trackerExpectedHitsOuter().numberOfLostHits();
      
      double mvaBDTG = tmvaReader_->EvaluateMVA("BDTG");
      if(mvaBDTG < minBDTG_)continue;
      
      
      TrackCandidate mergedTrack = merger_.merge(t1,t2);
      out_duplicateCandidates->push_back(mergedTrack);
      std::pair<Track,Track> trackPair(t1,t2);
      std::pair<TrackCandidate, std::pair<Track,Track> > cp(mergedTrack,trackPair);
      out_candidateMap->push_back(cp);
    }
  }
  iEvent.put(out_duplicateCandidates,"candidates");
  iEvent.put(out_candidateMap,"candidateMap");

}

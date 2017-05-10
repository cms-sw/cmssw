#ifndef RecoTauTag_RecoTau_PFRecoTauTagInfoAlgorithm_H
#define RecoTauTag_RecoTau_PFRecoTauTagInfoAlgorithm_H

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h" 
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Math/GenVector/VectorUtil.h" 

#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

class  PFRecoTauTagInfoAlgorithm  {
 public:
  PFRecoTauTagInfoAlgorithm(){}
  PFRecoTauTagInfoAlgorithm(const edm::ParameterSet&);
  ~PFRecoTauTagInfoAlgorithm(){}
  reco::PFTauTagInfo buildPFTauTagInfo(const reco::PFJetRef&,const std::vector<reco::PFCandidatePtr>&,const reco::TrackRefVector&,const reco::Vertex&); 
 private: 
  double ChargedHadrCand_tkminPt_;
  int ChargedHadrCand_tkminPixelHitsn_;
  int ChargedHadrCand_tkminTrackerHitsn_;
  double ChargedHadrCand_tkmaxipt_;
  double ChargedHadrCand_tkmaxChi2_;
  double ChargedHadrCand_tkPVmaxDZ_;
  // 
  double NeutrHadrCand_HcalclusMinEt_;
  // 
  double GammaCand_EcalclusMinEt_;
  double ChargedHadronsAssociationCone_;
  // 
  double tkminPt_;
  int tkminPixelHitsn_;
  int tkminTrackerHitsn_;
  double tkmaxipt_;
  double tkmaxChi2_;
  double tkPVmaxDZ_;
  // 
  bool UsePVconstraint_;
};
#endif 


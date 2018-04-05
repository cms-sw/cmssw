#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoAlgorithm.h"

using namespace reco;

PFRecoTauTagInfoAlgorithm::PFRecoTauTagInfoAlgorithm(const edm::ParameterSet& parameters){
  // parameters of the considered charged hadr. PFCandidates, based on their rec. tk properties :
  ChargedHadronsAssociationCone_      = parameters.getParameter<double>("ChargedHadrCand_AssociationCone");
  ChargedHadrCand_tkminPt_            = parameters.getParameter<double>("ChargedHadrCand_tkminPt");
  ChargedHadrCand_tkminPixelHitsn_    = parameters.getParameter<int>("ChargedHadrCand_tkminPixelHitsn");
  ChargedHadrCand_tkminTrackerHitsn_  = parameters.getParameter<int>("ChargedHadrCand_tkminTrackerHitsn");
  ChargedHadrCand_tkmaxipt_           = parameters.getParameter<double>("ChargedHadrCand_tkmaxipt");
  ChargedHadrCand_tkmaxChi2_          = parameters.getParameter<double>("ChargedHadrCand_tkmaxChi2");
  // parameters of the considered neutral hadr. PFCandidates, based on their rec. HCAL clus. properties :
  NeutrHadrCand_HcalclusMinEt_         = parameters.getParameter<double>("NeutrHadrCand_HcalclusMinEt");
  // parameters of the considered gamma PFCandidates, based on their rec. ECAL clus. properties :
  GammaCand_EcalclusMinEt_             = parameters.getParameter<double>("GammaCand_EcalclusMinEt");
  // parameters of the considered rec. Tracks (these ones catched through a JetTracksAssociation object, not through the charged hadr. PFCandidates inside the PFJet ; the motivation for considering them is the need for checking that a selection by the charged hadr. PFCandidates is equivalent to a selection by the rec. Tracks.) :
  tkminPt_                            = parameters.getParameter<double>("tkminPt");
  tkminPixelHitsn_                    = parameters.getParameter<int>("tkminPixelHitsn");
  tkminTrackerHitsn_                  = parameters.getParameter<int>("tkminTrackerHitsn");
  tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
  tkmaxChi2_                          = parameters.getParameter<double>("tkmaxChi2");
  //
  UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");
  ChargedHadrCand_tkPVmaxDZ_          = parameters.getParameter<double>("ChargedHadrCand_tkPVmaxDZ");
  tkPVmaxDZ_                          = parameters.getParameter<double>("tkPVmaxDZ");
}

PFTauTagInfo PFRecoTauTagInfoAlgorithm::buildPFTauTagInfo(const PFJetRef& thePFJet,const std::vector<reco::PFCandidatePtr>& thePFCandsInEvent, const TrackRefVector& theTracks,const Vertex& thePV) const {
  PFTauTagInfo resultExtended;
  resultExtended.setpfjetRef(thePFJet);

  std::vector<reco::PFCandidatePtr> thePFCands;
  const float jetPhi = (*thePFJet).phi();
  const float jetEta = (*thePFJet).eta();
  auto dr2 = [jetPhi,jetEta](float phi, float eta) { return reco::deltaR2(jetEta,jetPhi,eta,phi);};
  for (auto iPFCand : thePFCandsInEvent){
    float delta = dr2((*iPFCand).phi(),(*iPFCand).eta());
    if (delta < ChargedHadronsAssociationCone_*ChargedHadronsAssociationCone_)  thePFCands.push_back(iPFCand);
  }
  bool pvIsFake = (thePV.z() < -500.);

  std::vector<reco::PFCandidatePtr> theFilteredPFChargedHadrCands;
  if (UsePVconstraint_ && !pvIsFake) theFilteredPFChargedHadrCands=TauTagTools::filteredPFChargedHadrCands(thePFCands,ChargedHadrCand_tkminPt_,ChargedHadrCand_tkminPixelHitsn_,ChargedHadrCand_tkminTrackerHitsn_,ChargedHadrCand_tkmaxipt_,ChargedHadrCand_tkmaxChi2_,ChargedHadrCand_tkPVmaxDZ_, thePV, thePV.z());
  else theFilteredPFChargedHadrCands=TauTagTools::filteredPFChargedHadrCands(thePFCands,ChargedHadrCand_tkminPt_,ChargedHadrCand_tkminPixelHitsn_,ChargedHadrCand_tkminTrackerHitsn_,ChargedHadrCand_tkmaxipt_,ChargedHadrCand_tkmaxChi2_, thePV);
  resultExtended.setPFChargedHadrCands(theFilteredPFChargedHadrCands);
  resultExtended.setPFNeutrHadrCands(TauTagTools::filteredPFNeutrHadrCands(thePFCands,NeutrHadrCand_HcalclusMinEt_));
  resultExtended.setPFGammaCands(TauTagTools::filteredPFGammaCands(thePFCands,GammaCand_EcalclusMinEt_));

  TrackRefVector theFilteredTracks;
  if (UsePVconstraint_ && !pvIsFake) theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,tkPVmaxDZ_,thePV, thePV.z());
  else theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkminPixelHitsn_,tkminTrackerHitsn_,tkmaxipt_,tkmaxChi2_,thePV);
  resultExtended.setTracks(theFilteredTracks);

  return resultExtended;
}


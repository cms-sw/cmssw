#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoAlgorithm.h"

PFRecoTauTagInfoAlgorithm::PFRecoTauTagInfoAlgorithm(const ParameterSet& parameters){
  // parameters of the considered charged hadr. PFCandidates, based on their rec. tk properties :
  ChargedHadronsAssociationCone_      = parameters.getParameter<double>("ChargedHadrCand_AssociationCone");

  ChargedHadrCand_tkminPt_            = parameters.getParameter<double>("ChargedHadrCand_tkminPt");
  ChargedHadrCand_tkmaxipt_           = parameters.getParameter<double>("ChargedHadrCand_tkmaxipt");
    // parameters of the considered neutral hadr. PFCandidates, based on their rec. HCAL clus. properties : 
  NeutrHadrCand_HcalclusminE_         = parameters.getParameter<double>("NeutrHadrCand_HcalclusminE");
  // parameters of the considered gamma PFCandidates, based on their rec. ECAL clus. properties :
  GammaCand_EcalclusminE_             = parameters.getParameter<double>("GammaCand_EcalclusminE");
  // parameters of the considered rec. Tracks (these ones catched through a JetTracksAssociation object, not through the charged hadr. PFCandidates inside the PFJet ; the motivation for considering them is the need for checking that a selection by the charged hadr. PFCandidates is equivalent to a selection by the rec. Tracks.) :
  tkminPt_                            = parameters.getParameter<double>("tkminPt");
  tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
  // 
  UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");  
  ChargedHadrCand_tkPVmaxDZ_          = parameters.getParameter<double>("ChargedHadrCand_tkPVmaxDZ");
}
PFTauTagInfo PFRecoTauTagInfoAlgorithm::buildPFTauTagInfo(const PFJetRef& thePFJet,const PFCandidateRefVector& thePFCandsInEvent, const TrackRefVector& theTracks,const Vertex& thePV){
  PFTauTagInfo resultExtended;
  resultExtended.setpfjetRef(thePFJet);
  
  PFCandidateRefVector thePFCands;
  for (PFCandidateRefVector::const_iterator iPFCand=thePFCandsInEvent.begin();iPFCand!=thePFCandsInEvent.end();iPFCand++){
    double delta = ROOT::Math::VectorUtil::DeltaR((*thePFJet).p4().Vect(), (*iPFCand)->p4().Vect());
    if (delta < ChargedHadronsAssociationCone_)  thePFCands.push_back(*iPFCand);   
  }
  
  
  PFCandidateRefVector theFilteredPFChargedHadrCands;
  if (UsePVconstraint_) theFilteredPFChargedHadrCands=TauTagTools::filteredPFChargedHadrCands(thePFCands,ChargedHadrCand_tkminPt_,ChargedHadrCand_tkmaxipt_,ChargedHadrCand_tkPVmaxDZ_, thePV, thePV.z());
  else theFilteredPFChargedHadrCands=TauTagTools::filteredPFChargedHadrCands(thePFCands,ChargedHadrCand_tkminPt_,ChargedHadrCand_tkmaxipt_,thePV);
  resultExtended.setPFChargedHadrCands(theFilteredPFChargedHadrCands);
  resultExtended.setPFNeutrHadrCands(TauTagTools::filteredPFNeutrHadrCands(thePFCands,NeutrHadrCand_HcalclusminE_));
  resultExtended.setPFGammaCands(TauTagTools::filteredPFGammaCands(thePFCands,GammaCand_EcalclusminE_));
  
  TrackRefVector theFilteredTracks;
  if (UsePVconstraint_) theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkmaxipt_,ChargedHadrCand_tkPVmaxDZ_,thePV, thePV.z());
  else theFilteredTracks=TauTagTools::filteredTracks(theTracks,tkminPt_,tkmaxipt_,thePV);
  resultExtended.setTracks(theFilteredTracks);

  return resultExtended; 
}


#ifndef PFRecoTauTagInfoAlgorithm_H
#define PFRecoTauTagInfoAlgorithm_H

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h" 
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Math/GenVector/VectorUtil.h" 

using namespace std;
using namespace reco;
using namespace edm;

class  PFRecoTauTagInfoAlgorithm  {
 public:
  PFRecoTauTagInfoAlgorithm(){}  
  PFRecoTauTagInfoAlgorithm(const ParameterSet& parameters):PFChargedHadrCand_codenumber_(1),PFNeutrHadrCand_codenumber_(5),PFGammaCand_codenumber_(4),PFRecTrack_codenumber_(1),PFRecECALClus_codenumber_(4),PFRecHCALClus_codenumber_(5){
    // parameters of the considered charged hadr. PFCand.'s, based on their rec. tk properties :
    ChargedHadrCand_tkminPt_            = parameters.getParameter<double>("ChargedHadrCand_tkminPt");
    ChargedHadrCand_tkminPixelHitsn_    = parameters.getParameter<int>("ChargedHadrCand_tkminPixelHitsn");
    ChargedHadrCand_tkminTrackerHitsn_  = parameters.getParameter<int>("ChargedHadrCand_tkminTrackerHitsn");
    ChargedHadrCand_tkmaxipt_           = parameters.getParameter<double>("ChargedHadrCand_tkmaxipt");
    ChargedHadrCand_tkmaxChi2_          = parameters.getParameter<double>("ChargedHadrCand_tkmaxChi2");
    ChargedHadrCand_tktorefpointDZ_     = parameters.getParameter<double>("ChargedHadrCand_tktorefpointDZ");
    // parameters of the considered neutral hadr. PFCand.'s, based on their rec. HCAL clus. properties : 
    NeutrHadrCand_HcalclusminE_         = parameters.getParameter<double>("NeutrHadrCand_HcalclusminE");
    // parameters of the considered gamma PFCand.'s, based on their rec. ECAL clus. properties :
    GammaCand_EcalclusminE_             = parameters.getParameter<double>("GammaCand_EcalclusminE");
    // parameters of the considered rec. tk's (these ones catched through a JetTracksAssociator object, not through the charged hadr. PFCand.'s inside the PFJet) :
    tkminPt_                            = parameters.getParameter<double>("tkminPt");
    tkminPixelHitsn_                    = parameters.getParameter<int>("tkminPixelHitsn");
    tkminTrackerHitsn_                  = parameters.getParameter<int>("tkminTrackerHitsn");
    tkmaxipt_                           = parameters.getParameter<double>("tkmaxipt");
    tkmaxChi2_                          = parameters.getParameter<double>("tkmaxChi2");
    tktorefpointDZ_                     = parameters.getParameter<double>("tktorefpointDZ");
    // 
    UsePVconstraint_                    = parameters.getParameter<bool>("UsePVconstraint");  
  }
  ~PFRecoTauTagInfoAlgorithm(){}
  PFTauTagInfo tag(const PFJetRef&,const TrackRefVector&,const Vertex&); 
 private:
  PFCandidateRefVector filteredPFChargedHadrCands(PFCandidateRefVector thePFCands,double ChargedHadrCand_tkminPt,int ChargedHadrCand_tkminPixelHitsn,int ChargedHadrCand_tkminTrackerHitsn,double ChargedHadrCand_tkmaxipt,double ChargedHadrCand_tkmaxChi2,double ChargedHadrCand_tktorefpointDZ,bool UsePVconstraint,double PVtx_Z);
  PFCandidateRefVector filteredPFNeutrHadrCands(PFCandidateRefVector thePFCands,double NeutrHadrCand_HcalclusminEt);
  PFCandidateRefVector filteredPFGammaCands(PFCandidateRefVector thePFCands,double GammaCand_EcalclusminEt);
  TrackRefVector filteredTracks(TrackRefVector theTracks,double tkminPt,int tkminPixelHitsn,int tkminTrackerHitsn,double tkmaxipt,double tkmaxChi2,double tktorefpointDZ,bool UsePVconstraint,double PVtx_Z);
  // 
  double ChargedHadrCand_tkminPt_;
  int ChargedHadrCand_tkminPixelHitsn_;
  int ChargedHadrCand_tkminTrackerHitsn_;
  double ChargedHadrCand_tkmaxipt_;
  double ChargedHadrCand_tkmaxChi2_;
  double ChargedHadrCand_tktorefpointDZ_;
  // 
  double NeutrHadrCand_HcalclusminE_;
  // 
  double GammaCand_EcalclusminE_;
  // 
  double tkminPt_;
  int tkminPixelHitsn_;
  int tkminTrackerHitsn_;
  double tkmaxipt_;
  double tkmaxChi2_;
  double tktorefpointDZ_;
  // 
  bool UsePVconstraint_;
  //
  int PFChargedHadrCand_codenumber_;
  int PFNeutrHadrCand_codenumber_;
  int PFGammaCand_codenumber_;
  
  int PFRecTrack_codenumber_;
  int PFRecECALClus_codenumber_;
  int PFRecHCALClus_codenumber_;
};
#endif 


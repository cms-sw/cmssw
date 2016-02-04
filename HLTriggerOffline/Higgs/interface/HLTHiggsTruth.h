#ifndef HLTHIGGSTRUTH_H
#define HLTHIGGSTRUTH_H

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include <vector>
#include <map>
#include "TROOT.h"
#include "TChain.h"
#include "TVector3.h"
#include "TLorentzVector.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <DataFormats/HepMCCandidate/interface/GenParticle.h>

#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include <DataFormats/METReco/interface/CaloMET.h> 
#include <DataFormats/METReco/interface/CaloMETFwd.h> 
#include <DataFormats/METReco/interface/MET.h> 
#include <DataFormats/METReco/interface/METFwd.h> 



#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"


#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"


typedef std::vector<std::string> MyStrings;

/** \class HLTHiggsTruth
  *  
  * $Date: November 2006
  * $Revision: 
  * \author P. Bargassa - Rice U.
  */
class HLTHiggsTruth {
public:
  HLTHiggsTruth(); 
 
 

  void setup(const edm::ParameterSet& pSet, TTree* tree);

  /** Analyze the Data */



 // void analyzeHWW2l(const reco::CandidateView& mctruth,const reco::MuonCollection& muonHandle, const GsfElectronCollection& electronHandle, TTree* tree);
  
   void analyzeHWW2l(const reco::CandidateView& mctruth,const reco::CaloMETCollection&
   caloMet, const reco::TrackCollection& Tracks, const reco::MuonCollection& muonHandle, const reco::GsfElectronCollection& electronHandle, TTree* tree);
  
  
  void analyzeHZZ4l(const reco::CandidateView& mctruth,const reco::MuonCollection& muonHandle, const reco::GsfElectronCollection& electronHandle, TTree* tree);
  void analyzeHgg(const reco::CandidateView& mctruth,const reco::PhotonCollection& photonHandle,TTree* tree); 
  void analyzeH2tau(const reco::CandidateView& mctruth,TTree* tree);  
  void analyzeHtaunu(const reco::CandidateView& mctruth,TTree* tree);  
  void analyzeA2mu(const reco::CandidateView& mctruth,TTree* tree);  
  void analyzeHinv(const reco::CandidateView& mctruth,TTree* tree);
  
  /* study leptonic tau decays*/
  void LeptonicTauDecay(const reco::Candidate& tau, bool& elecdec, bool&
  muondec);  
     

  inline bool decision() const {return isvisible;};
  
 /* inline bool MuonChannel() const {return isMuonDecay;};  // muons in final state
  inline bool ElecChannel() const {return isElecDecay;};  // electrons in final state
  inline bool ElecMuchannel() const {return isEMuDecay;};
 
  
  inline bool MuonChannel_acc() const {return isMuonDecay_acc;}; //within acceptance cuts
  inline bool ElecChannel_acc() const {return isElecDecay_acc;};
  inline bool ElecMuChannel_acc() const {return isEMuDecay_acc;}; */
  inline bool PhotonChannel_acc() const {return isPhotonDecay_acc;};
  inline bool TauChannel_acc() const {return isTauDecay_acc;};
  
  
  inline bool MuonChannel_recoacc() const {return isMuonDecay_recoacc;}; //within acceptance +reco cuts
  inline bool ElecChannel_recoacc() const {return isElecDecay_recoacc;};
  inline bool ElecMuChannel_recoacc() const {return isEMuDecay_recoacc;}; 
  inline bool PhotonChannel_recoacc() const {return isPhotonDecay_recoacc;};
  inline bool TauChannel_recoacc() const {return isTauDecay_recoacc;};
  inline bool decision_reco() const {return isvisible_reco;};
  
  
  inline double ptMuon1() const {return ptMuMax;};
  inline double ptMuon2() const {return ptMuMin;};
  inline double etaMuon1() const {return etaMuMax;};
  inline double etaMuon2() const {return etaMuMin;};
  inline double ptElectron1() const {return ptElMax;};
  inline double ptElectron2() const {return ptElMin;};
  inline double etaElectron1() const {return etaElMax;};
  inline double etaElectron2() const {return etaElMin;};
/*  inline double ptPhoton1() const {return ptPhMax;};
  inline double ptPhoton2() const {return ptPhMin;};
  inline double etaPhoton1() const {return etaPhMax;};
  inline double etaPhoton2() const {return etaPhMin;};*/
  inline double ptTau1() const {return ptTauMax;};
  inline double etaTau1() const {return etaTauMax;};
  
 /*  inline int evtype() const {return ev_type;};*/
  
 /* inline std::vector<double> ptgen_() const {return ptgen;};
  inline std::vector<double> ptreco_() const {return ptreco;};
 
 inline int ngenpart_() const {return ngenpart;};
 inline int ngenph_() const{return ngenph;};*/
 
 
 inline reco::Photon photon1_() const {return Photon1;};
 inline reco::Photon photon2_() const {return Photon2;};
 
 inline reco::Muon muon1_() const {return Muon1;};
 inline reco::Muon muon2_() const {return Muon2;};
 inline reco::Muon muon3_() const {return Muon3;};
 inline reco::Muon muon4_() const {return Muon4;};
 
/* inline reco::Particle genmuon1_() const {return genMuon1;};
 inline reco::Particle genelectron1_() const {return genElectron1;};*/
 
 inline reco::GsfElectron electron1_() const {return Electron1;};
 inline reco::GsfElectron electron2_() const {return Electron2;};
 inline reco::GsfElectron electron3_() const {return Electron3;};
 inline reco::GsfElectron electron4_() const {return Electron4;};
 
 inline double met_hwwdimu() const {return met_hwwdimu_ ;};
 inline double met_hwwdiel() const {return met_hwwdiel_ ;};
 inline double met_hwwemu()  const {return met_hwwemu_ ;};
 

private:

  // Tree variables
/*  float *mcpid, *mcvx, *mcvy, *mcvz, *mcpt;
//  bool isvisible_WW, isvisible_ZZ, isvisible_gg, isvisible_2tau, isvisible_taunu,
//       isvisible_2mu, isvisible_inv,isvisible;*/
  bool isvisible;
/*  bool isMuonDecay;
  bool isElecDecay;
  bool isEMuDecay;
  bool isMuonDecay_acc;
  bool isElecDecay_acc;
  bool isEMuDecay_acc;
  bool isPhotonDecay;*/
  bool isPhotonDecay_acc;
  bool isTauDecay;
  bool isTauDecay_acc;
  
  
  bool isMuonDecay_recoacc;
  bool isElecDecay_recoacc;
  bool isEMuDecay_recoacc; 
  bool isPhotonDecay_recoacc;
  bool isTauDecay_recoacc;
  
  bool isvisible_reco;
  
/*  int ev_type; */
  
  
  double ptMuMax, ptMuMin;
  double ptElMax, ptElMin;
/*  double ptPhMax, ptPhMin;*/
  double ptTauMax;
  double etaMuMax, etaMuMin;
  double etaElMax, etaElMin;
  double etaPhMax, etaPhMin;
  double etaTauMax;
 
  
  double PtElFromTau, PtMuFromTau;
  double EtaElFromTau, EtaMuFromTau;
  
  double met_hwwdimu_;
  double met_hwwdiel_;
  double met_hwwemu_;
  
 /* std::vector<double> ptgen ; // etagen;
  std::vector<double> ptreco;// etareco;*/
  
   reco::Photon Photon1;
   reco::Photon Photon2;
   
   reco::Muon Muon1;
   reco::Muon Muon2;
   reco::Muon Muon3;
   reco::Muon Muon4;
   
   reco::GsfElectron Electron1;
   reco::GsfElectron Electron2;
   reco::GsfElectron Electron3;
   reco::GsfElectron Electron4;
   
 /*  reco::Particle genMuon1;
   reco::Particle genElectron1;
  
  
 // double ptgen,ptreco;
  int n_gen, n_match;
  int ngenpart, ngenph; */
  
  // input variables
  bool _Monte,_Debug;

};

#endif

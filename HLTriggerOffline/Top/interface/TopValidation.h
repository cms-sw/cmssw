
#ifndef HLTriggerOffline_TopValidation_H
#define HLTriggerOffline_TopValidation_H 


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h" 
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h" 

//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <DataFormats/HepMCCandidate/interface/GenParticle.h>
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"



#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "HLTrigger/HLTfilters/interface/HLTHighLevel.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"




#include "TH1F.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <math.h> 
#include <TMath.h>
#include "TFile.h"



class TH1F;

//
// class decleration
//

class TopValidation : public edm::EDAnalyzer {
public:
  explicit TopValidation(const edm::ParameterSet&);
  ~TopValidation();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 
 // edm::Service<TFileService> fs;
  
  
  // ----------member data ---------------------------
  
  string fOutputFileName ;
  
std::string outFile_, outputFileName;

bool outputMEsInRootFile;



 edm::ParameterSet parameters;

 DQMStore* dbe;
 ///// eta and pt histos ////
 
 //  semimu events
  MonitorElement*     eta_trig_off_mu;
  MonitorElement*     eta_off_mu;
  MonitorElement*     pt_trig_off_mu;
  MonitorElement*     pt_off_mu;
 

  //  dimu events
  MonitorElement*     eta_trig_off_dimu0;
  MonitorElement*     eta_off_dimu0;
  MonitorElement*     pt_trig_off_dimu0;
  MonitorElement*     pt_off_dimu0;
 
  MonitorElement*     eta_trig_off_dimu1;
  MonitorElement*     eta_off_dimu1;
  MonitorElement*     pt_trig_off_dimu1;
  MonitorElement*     pt_off_dimu1;

//  semiel events 
  MonitorElement*     eta_trig_off_el;
  MonitorElement*     eta_off_el;
  MonitorElement*     pt_trig_off_el;
  MonitorElement*     pt_off_el;
  MonitorElement*     pt_trig_off_el_ni;
  MonitorElement*     pt_trig_off_el_li;
  MonitorElement*     eta_trig_off_el_ni;
  MonitorElement*     eta_trig_off_el_li;

  //diel events 
  MonitorElement*     eta_trig_off_diel0;
  MonitorElement*     eta_off_diel0;
  MonitorElement*     pt_trig_off_diel0;
  MonitorElement*     pt_off_diel0;

  MonitorElement*     eta_trig_off_diel1;
  MonitorElement*     eta_off_diel1;
  MonitorElement*     pt_trig_off_diel1;
  MonitorElement*     pt_off_diel1;
 
  //  emu events
  MonitorElement*     eta_trig_off_emu_muon;
  MonitorElement*     eta_off_emu_muon;
  MonitorElement*     pt_trig_off_emu_muon;
  MonitorElement*     pt_off_emu_muon;
 
  MonitorElement*     eta_trig_off_emu_electron;
  MonitorElement*     eta_off_emu_electron;
  MonitorElement*     pt_trig_off_emu_electron;
  MonitorElement*     pt_off_emu_electron;
  
  
 
 ///// histos event type///// 
  TH1F*     Nttbar;
  TH1F*     Ndilep;
  TH1F*     Nsemilep;
  TH1F*     Nhadronic;
  TH1F*     Nsemimu;
  TH1F*     Nsemiel;
  TH1F*     Nsemitau;
  TH1F*     Ndimu;
  TH1F*     Ndiel;
  TH1F*     Nemu;
  TH1F*     Nditau;
  TH1F*     Ntaumu;
  TH1F*     Ntauel;
  
 
  
  
  /////histos #events passing acceptance cuts ////
  TH1F*     acceptmu;
  TH1F*     acceptel;
  TH1F*     acceptdimu;
  TH1F*     acceptdiel;
  TH1F*     acceptemu;
  
  TH1F*     acceptmufromtau;
  TH1F*     acceptelfromtau;
  TH1F*     acceptdimufromtau;
  TH1F*     acceptdielfromtau;
  TH1F*     acceptemufromtau;
 
 ///HISTOS EFFICIENCIES 
  
  /// for semimu events
  TH1F*     Noffmu;                       // #events passing offline selection
  
  TH1F*     hlt1muon_off_semimu;          // #events passing trigger after offline
  TH1F*     hlt1muon_semimu;              // #events passing trigger after acceptance
  TH1F*     off_hlt1muon_semimu;          // #events passing offline after trigger
  
  TH1F*     hlt1muoniso_semimu;
  TH1F*     off_hlt1muoniso_semimu;
  TH1F*     hlt1muoniso_off_semimu;
  
  TH1F*     hlt1elecrelax_semimu;
  TH1F*     off_hlt1elecrelax_semimu;
  TH1F*     hlt1elecrelax_off_semimu;
  
  TH1F*     hlt1elec_semimu;
  TH1F*     off_hlt1elec_semimu;
  TH1F*     hlt1elec_off_semimu;
  
  TH1F*     hlt2muon_semimu;
  TH1F*     off_hlt2muon_semimu;
  TH1F*     hlt2muon_off_semimu;
  
  TH1F*     hlt2elec_semimu;
  TH1F*     off_hlt2elec_semimu;
  TH1F*     hlt2elec_off_semimu;
   
  TH1F*     hltelecmu_semimu;
  TH1F*     off_hltelecmu_semimu;
  TH1F*     hltelecmu_off_semimu;
  
  TH1F*     hltelecmurelax_semimu;
  TH1F*     off_hltelecmurelax_semimu;
  TH1F*     hltelecmurelax_off_semimu;
  
  TH1F*     hlt1jet_semimu;
  TH1F*     off_hlt1jet_semimu;
  TH1F*     hlt1jet_off_semimu;
  
  //----- new paths
  
  TH1F*     hlt4jet30_semimu;
  TH1F*     hlt4jet30_off_semimu;
  TH1F*     hlt1elec15_li_semimu;
  TH1F*     hlt1elec15_li_off_semimu;
  TH1F*     hlt1elec15_ni_semimu;
  TH1F*     hlt1elec15_ni_off_semimu;
  
  
  ///// for semiel events
   
  TH1F*     Noffel;
  
  TH1F*     hlt1muon_off_semiel;
  TH1F*     hlt1muon_semiel;
  TH1F*     off_hlt1muon_semiel;
  
  TH1F*     hlt1muoniso_semiel;
  TH1F*     off_hlt1muoniso_semiel;
  TH1F*     hlt1muoniso_off_semiel;
  
  TH1F*     hlt1elecrelax_semiel;
  TH1F*     off_hlt1elecrelax_semiel;
  TH1F*     hlt1elecrelax_off_semiel;
  
  TH1F*     hlt1elec_semiel;
  TH1F*     off_hlt1elec_semiel;
  TH1F*     hlt1elec_off_semiel;
  
  TH1F*     hlt2muon_semiel;
  TH1F*     off_hlt2muon_semiel;
  TH1F*     hlt2muon_off_semiel;
  
  TH1F*     hlt2elec_semiel;
  TH1F*     off_hlt2elec_semiel;
  TH1F*     hlt2elec_off_semiel;
   
  TH1F*     hltelecmu_semiel;
  TH1F*     off_hltelecmu_semiel;
  TH1F*     hltelecmu_off_semiel;
  
  TH1F*     hltelecmurelax_semiel;
  TH1F*     off_hltelecmurelax_semiel;
  TH1F*     hltelecmurelax_off_semiel;
  
  TH1F*     hlt1jet_semiel;
  TH1F*     off_hlt1jet_semiel;
  TH1F*     hlt1jet_off_semiel;
  
  //----- new paths
  
  TH1F*     hlt4jet30_semiel;
  TH1F*     hlt4jet30_off_semiel;
  TH1F*     hlt1elec15_li_semiel;
  TH1F*     hlt1elec15_li_off_semiel;
  TH1F*     hlt1elec15_ni_semiel;
  TH1F*     hlt1elec15_ni_off_semiel;
  
  
   
   /// for dimu events
   
  TH1F*     Noffdimu;
  
  TH1F*     hlt1muon_off_dimu;
  TH1F*     hlt1muon_dimu;
  TH1F*     off_hlt1muon_dimu;
  

  TH1F*     hlt1muoniso_dimu;
  TH1F*     off_hlt1muoniso_dimu;
  TH1F*     hlt1muoniso_off_dimu;
  
  TH1F*     hlt1elecrelax_dimu;
  TH1F*     off_hlt1elecrelax_dimu;
  TH1F*     hlt1elecrelax_off_dimu;
  
  TH1F*     hlt1elec_dimu;
  TH1F*     off_hlt1elec_dimu;
  TH1F*     hlt1elec_off_dimu;
  
  TH1F*     hlt2muon_dimu;
  TH1F*     off_hlt2muon_dimu;
  TH1F*     hlt2muon_off_dimu;
  
  TH1F*     hlt2elec_dimu;
  TH1F*     off_hlt2elec_dimu;
  TH1F*     hlt2elec_off_dimu;
   
  TH1F*     hltelecmu_dimu;
  TH1F*     off_hltelecmu_dimu;
  TH1F*     hltelecmu_off_dimu;
  
  TH1F*     hltelecmurelax_dimu;
  TH1F*     off_hltelecmurelax_dimu;
  TH1F*     hltelecmurelax_off_dimu;
  
  TH1F*     hlt1jet_dimu;
  TH1F*     off_hlt1jet_dimu;
  TH1F*     hlt1jet_off_dimu;
  
  //----- new paths
  
  TH1F*     hlt4jet30_dimu;
  TH1F*     hlt4jet30_off_dimu;
  TH1F*     hlt1elec15_li_dimu;
  TH1F*     hlt1elec15_li_off_dimu;
  TH1F*     hlt1elec15_ni_dimu;
  TH1F*     hlt1elec15_ni_off_dimu;
  
  
  
  //// for diel events
   
  TH1F*     Noffdiel;
  
  TH1F*     hlt1muon_off_diel;
  TH1F*     hlt1muon_diel;
  TH1F*     off_hlt1muon_diel;
  

  TH1F*     hlt1muoniso_diel;
  TH1F*     off_hlt1muoniso_diel;
  TH1F*     hlt1muoniso_off_diel;
  
  TH1F*     hlt1elecrelax_diel;
  TH1F*     off_hlt1elecrelax_diel;
  TH1F*     hlt1elecrelax_off_diel;
  
  TH1F*     hlt1elec_diel;
  TH1F*     off_hlt1elec_diel;
  TH1F*     hlt1elec_off_diel;
  
  TH1F*     hlt2muon_diel;
  TH1F*     off_hlt2muon_diel;
  TH1F*     hlt2muon_off_diel;
  
  TH1F*     hlt2elec_diel;
  TH1F*     off_hlt2elec_diel;
  TH1F*     hlt2elec_off_diel;
   
  TH1F*     hltelecmu_diel;
  TH1F*     off_hltelecmu_diel;
  TH1F*     hltelecmu_off_diel;
  
  TH1F*     hltelecmurelax_diel;
  TH1F*     off_hltelecmurelax_diel;
  TH1F*     hltelecmurelax_off_diel;
  
  TH1F*     hlt1jet_diel;
  TH1F*     off_hlt1jet_diel;
  TH1F*     hlt1jet_off_diel;
  
  //----- new paths
  
  TH1F*     hlt4jet30_diel;
  TH1F*     hlt4jet30_off_diel;
  TH1F*     hlt1elec15_li_diel;
  TH1F*     hlt1elec15_li_off_diel;
  TH1F*     hlt1elec15_ni_diel;
  TH1F*     hlt1elec15_ni_off_diel;
  
  
  //// for emu events
   
  TH1F*     Noffemu;
  
  TH1F*     hlt1muon_off_emu;
  TH1F*     hlt1muon_emu;
  TH1F*     off_hlt1muon_emu;
  

  TH1F*     hlt1muoniso_emu;
  TH1F*     off_hlt1muoniso_emu;
  TH1F*     hlt1muoniso_off_emu;
  
  TH1F*     hlt1elecrelax_emu;
  TH1F*     off_hlt1elecrelax_emu;
  TH1F*     hlt1elecrelax_off_emu;
  
  TH1F*     hlt1elec_emu;
  TH1F*     off_hlt1elec_emu;
  TH1F*     hlt1elec_off_emu;
  
  TH1F*     hlt2muon_emu;
  TH1F*     off_hlt2muon_emu;
  TH1F*     hlt2muon_off_emu;
  
  TH1F*     hlt2elec_emu;
  TH1F*     off_hlt2elec_emu;
  TH1F*     hlt2elec_off_emu;
   
  TH1F*     hltelecmu_emu;
  TH1F*     off_hltelecmu_emu;
  TH1F*     hltelecmu_off_emu;
  
  TH1F*     hltelecmurelax_emu;
  TH1F*     off_hltelecmurelax_emu;
  TH1F*     hltelecmurelax_off_emu;
  
  TH1F*     hlt1jet_emu;
  TH1F*     off_hlt1jet_emu;
  TH1F*     hlt1jet_off_emu;
  
  TH1F*   OR_emu;
  TH1F*   OR_off_emu;
  TH1F*   OR_emu_li;
  TH1F*   OR_emu_ni;
  TH1F*   OR_off_emu_li;
  TH1F*   OR_off_emu_ni;
  
  //----- new paths
  
  TH1F*     hlt4jet30_emu;
  TH1F*     hlt4jet30_off_emu;
  TH1F*     hlt1elec15_li_emu;
  TH1F*     hlt1elec15_li_off_emu;
  TH1F*     hlt1elec15_ni_emu;
  TH1F*     hlt1elec15_ni_off_emu;
  
  
 
  
    
  unsigned int nEvents;
  unsigned int nAccepted;
  unsigned int n_top;
  unsigned int n_w;
  unsigned int n_z;

  
  unsigned int dilepEvent,semilepEvent,hadronicEvent;
  unsigned int dimuEvent,dielEvent,emuEvent,muEvent,tauEvent,elecEvent,taumuEvent,tauelEvent,ditauEvent;
 
 unsigned int munoiso;
  
  TString name51,name50,name33,name0,name32,name35,name92,name93,name53;
  TString name52,name34,name54,name36,name94,name1;
  
  
  void efficiencies (TH1F *select, TH1F *path1,TH1F *path2, TH1F *path3, TH1F *path4,
 TH1F* path5, TH1F *path6, TH1F *path7, TH1F *path8, TH1F *path9, TH1F *path10,
 TH1F* path11, TH1F* path12);
 
 
 
  edm::InputTag inputTag_;
  edm::TriggerNames triggerNames_;
  
  
  
  //Just a tag for better file organization
  std::string triggerTag_;

  //MonitorElements

  /*Trigger Bits for Tau and Reference Trigger*/
  MonitorElement *l1eteff;
  MonitorElement *l2eteff;
  MonitorElement *l25eteff;
  MonitorElement *l3eteff;

  MonitorElement *refEt;
  MonitorElement *refEta;

  MonitorElement *l1etaeff;
  MonitorElement *l2etaeff;
  MonitorElement *l25etaeff;
  MonitorElement *l3etaeff;

  MonitorElement *accepted_events;
  MonitorElement *accepted_events_matched;

  
  //HLT Path decisions
    bool HLT1MuonNonIso;            
    bool HLT2MuonNonIso ;               
    bool  HLT1MuonIso;                    
    bool HLT1ElectronRelaxed;            
    bool HLT2ElectronRelaxed;            
    bool HLTXElectronMuon;               
    bool HLT1Electron;                   
    bool HLTXElectronMuonRelaxed;       
    bool HLT1jet;    
    
    bool  HLT4jet30               ;
    bool  HLT1Electron15_NI       ;
    bool  HLT1Electron15_LI       ;                    
 
  
};


#endif

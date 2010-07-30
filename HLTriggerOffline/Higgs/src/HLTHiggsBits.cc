/** \class HLTHiggsBits
 *
 * See header file for documentation
 *
 *  $Date: 2010/05/03 16:39:07 $
 *  $Revision: 1.6 $
 *
 *  \author Mika Huhtinen
 *
 */

#include "HLTriggerOffline/Higgs/interface/HLTHiggsBits.h"
#include "HLTriggerOffline/Higgs/interface/HLTHiggsTruth.h"


#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/*#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
//#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
//#include "DataFormats/L1Trigger/interface/L1ParticleMap.h" */

#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include <iomanip>
#include <string>
#include <vector>

//
// constructors and destructor
//
HLTHiggsBits::HLTHiggsBits(const edm::ParameterSet& iConfig) :
  hlTriggerResults_ (iConfig.getParameter<edm::InputTag> ("HLTriggerResults")),
  mctruth_ (iConfig.getParameter<edm::InputTag> ("MCTruth")),
  n_channel_ (iConfig.getParameter<int>("Nchannel")),
  nEvents_(0),
  hlNames_(0),
  init_(false),
  histName(iConfig.getParameter<std::string>("histName")),
  hlt_bitnames(iConfig.getParameter<std::vector<std::string> >("hltBitNames")),
  hlt_bitnamesMu(iConfig.getParameter<std::vector<std::string> >("hltBitNamesMu")),
  hlt_bitnamesEg(iConfig.getParameter<std::vector<std::string> >("hltBitNamesEG")),
  hlt_bitnamesPh(iConfig.getParameter<std::vector<std::string> >("hltBitNamesPh")),
  hlt_bitnamesTau(iConfig.getParameter<std::vector<std::string> >("hltBitNamesTau")),
  triggerTag_(iConfig.getUntrackedParameter<std::string>("DQMFolder","HLT/Higgs")),
  outputFileName(iConfig.getParameter<std::string>("OutputFileName")),
  outputMEsInRootFile(iConfig.getParameter<bool>("OutputMEsInRootFile"))
 
  
{

   
   n_hlt_bits=hlt_bitnames.size();   // total paths
  
 
    n_hlt_bits_eg  = hlt_bitnamesEg.size();   // muon paths   
    n_hlt_bits_mu  = hlt_bitnamesMu.size();   // electron paths
    n_hlt_bits_ph  = hlt_bitnamesPh.size();  // photon paths
    n_hlt_bits_tau = hlt_bitnamesTau.size(); // tau paths
    
   
 /* 
  std::cout << "Number of bit names : " << n_hlt_bits << std::endl;
  if (n_hlt_bits>20) {
    std::cout << "TOO MANY BITS REQUESTED - TREATING ONLY FIRST 20" << std::endl;
    n_hlt_bits=20;
  }*/
  
  
  

// 1:H->ZZ->4l, 2:H->WW->2l, 3: H->gg, 4:qqh->2tau, 5:H+->taunu, 6:qqh->inv
// The proper channel number has to be set in the cff-file
 // std::cout << "Analyzing Higgs channel number " << n_channel_ << std::endl;

  // open the histogram file
 // m_file=0; // set to null
 // m_file=new TFile((histName+".root").c_str(),"RECREATE");
 // m_file->cd();
 // outfile.open((histName+".output").c_str());


  // Initialize the tree
  HltTree = 0;
  HltTree = new TTree("HltTree","");

 /* for (int i=0;i<n_hlt_bits;i++) {
    for (int j=0;j<n_hlt_bits+1;j++) {
      hlt_whichbit[0][i][j]=0;
      hlt_whichbit[1][i][j]=0;
      hlt_whichbit[2][i][j]=0;
    }
  }*/


  mct_analysis_.setup(iConfig, HltTree);

 // const int kMaxEvents = 50000;
 /* const int kMaxEvents = 5000000;
  hlt_nbits = new int[kMaxEvents];
  HltTree->Branch("NEventCount",&neventcount,"NEventCount/I");
  HltTree->Branch("HLT_nBits",hlt_nbits,"HLT_nBits[NEventCount]/I");*/
  

  
  //--------------------------------------------
  ///  histos pt, eta   reconstructed objects
  ///
  //--------------------------------------------
  
 
        dbe = edm::Service<DQMStore>().operator->();	
	dbe->setCurrentFolder(triggerTag_);

     //  dbe->setCurrentFolder("HLT/Higgs");
     
     
     if (n_channel_==2) {
      h_met_hwwdimu = dbe->book1D("caloMET_dimu","caloMET_dimu",50,0.0,150.0);
      h_met_hwwdiel = dbe->book1D("caloMET_diel","caloMET_diel",50,0.0,150.0);
      h_met_hwwemu  = dbe->book1D("caloMET_emu","caloMET_emu",50,0.0,150.0);
      
      }
     
     
     if (n_channel_==1 || n_channel_==2 || n_channel_==4){  // only for WW,ZZ, 2tau
  
   h_ptmu1 = dbe->book1D("Muon1Pt","Muon1Pt",50,0.0,150.0);
   h_ptmu2 = dbe->book1D("Muon2Pt","Muon2Pt",50,0.0,150.0);
   h_etamu1 = dbe->book1D("Muon1Eta","Muon1Eta",50,-2.5,2.5);
   h_etamu2 = dbe->book1D("Muon2Eta","Muon2Eta",50,-2.5,2.5);
        
   h_ptel1 = dbe->book1D("Electron1Pt","Electron1Pt",50,0.0,150.0);
   h_ptel2 = dbe->book1D("Electron2Pt","Electron2Pt",50,0.0,150.0);
   h_etael1 = dbe->book1D("Electron1Eta","Electron1Eta",50,-2.5,2.5);
   h_etael2 = dbe->book1D("Electron2Eta","Electron2Eta",50,-2.5,2.5);
   
   hlt_bitmu_hist_reco = dbe->book1D("muHLT","muHLT",hlt_bitnamesMu.size(),0.5,hlt_bitnamesMu.size()+0.5);
   h_mu_reco = dbe->book1D("MuonEvents","MuonEvents",hlt_bitnamesMu.size(),0.5,hlt_bitnamesMu.size()+0.5);
   
   hlt_bitel_hist_reco = dbe->book1D("elHLT","elHLT",hlt_bitnamesEg.size(),0.5,hlt_bitnamesEg.size()+0.5);
   h_el_reco = dbe->book1D("ElectronEvents","ElectronEvents",hlt_bitnamesEg.size(),0.5,hlt_bitnamesEg.size()+0.5);
    
   
   }
   
   if (n_channel_==1 || n_channel_==2){
         
   h_ptmu1_emu = dbe->book1D("Muon1Pt_EM","Muon1Pt_EM",50,0.0,150.0);
   h_ptel1_emu = dbe->book1D("Electron1Pt_EM","Electron1Pt_EM",50,0.0,150.0);
   h_etamu1_emu = dbe->book1D("Muon1Eta_EM","Muon1Eta_EM",50,-2.5,2.5);
   h_etael1_emu = dbe->book1D("Electron1Eta_EM","Electron1Eta_EM",50,-2.5,2.5);
   
   hlt_bitemu_hist_reco = dbe->book1D("emuHLT","emuHLT",hlt_bitnames.size(),0.5,hlt_bitnames.size()+0.5);
   h_emu_reco = dbe->book1D("EmuEvents","EmuEvents",hlt_bitnames.size(),0.5,hlt_bitnames.size()+0.5);
   
   
   }
 //  dbe->setCurrentFolder("HLT/Higgs/H2tau");
 
   if (n_channel_==3){  // only for Hgg
      
   h_ptph1 = dbe->book1D("Photon1Pt","Photon1Pt",50,0.0,200.0);
   h_ptph2 = dbe->book1D("Photon2Pt","Photon2Pt",50,0.0,200.0);
   h_etaph1 = dbe->book1D("Photon1Eta","Photon1Eta",50,-2.5,2.5);
   h_etaph2 = dbe->book1D("Photon2Eta","Photon2Eta",50,-2.5,2.5);
   
   hlt_bitph_hist_reco = dbe->book1D("phHLT","phHLT",hlt_bitnamesPh.size(),0.5,hlt_bitnamesPh.size()+0.5);
   h_ph_reco = dbe->book1D("PhotonEvents","PhotonEvents",hlt_bitnamesPh.size(),0.5,hlt_bitnamesPh.size()+0.5);
   
    
   }
   
   if (n_channel_==5){
   
  // dbe->setCurrentFolder("HLT/Higgs/Htaunu");   
  // h_pttau1 = dbe->book1D("Tau1Pt","Tau1Pt",50,0.0,500.0);  
  // h_etatau1 = dbe->book1D("Tau1Eta","Tau1Eta",50,-5.0,5.0);
  
   hlt_bittau_hist_gen = dbe->book1D("tauHLT","tauHLT",hlt_bitnamesTau.size(),0.5,hlt_bitnamesTau.size()+0.5);
   h_tau_gen = dbe->book1D("tauEvents","tauEvents",hlt_bitnamesTau.size(),0.5,hlt_bitnamesTau.size()+0.5);
    
   }
 
  //------------------------------------------------
  //
  //  Histos pt, eta RECO events firing HLT 
  //
  //---------------------------------------------------
  
     if (n_channel_==1 || n_channel_==2){
  
    for (int j=0;j<n_hlt_bits;j++) { 
     std::string histnameptmuem  = "Muon1Pt_EM_"+hlt_bitnames[j];
     std::string histnameetamuem = "Muon1Eta_EM_"+hlt_bitnames[j];
     std::string histnameptelem  = "Electron1Pt_EM_"+hlt_bitnames[j];
     std::string histnameetaelem = "Electron1Eta_EM_"+hlt_bitnames[j];
     h_ptmu1_emu_trig[j] = dbe->book1D((histnameptmuem).c_str(),(hlt_bitnames[j]+"ptmuon").c_str(),50,0.0,150.0); 
     h_etamu1_emu_trig[j] = dbe->book1D((histnameetamuem).c_str(),(hlt_bitnames[j]+"etamuon").c_str(),50,-2.5,2.5);
     
     h_ptel1_emu_trig[j] = dbe->book1D((histnameptelem).c_str(),(hlt_bitnames[j]+"ptelectron").c_str(),50,0.0,150.0); 
     h_etael1_emu_trig[j] = dbe->book1D((histnameetaelem).c_str(),(hlt_bitnames[j]+"etaelectron").c_str(),50,-2.5,2.5); 
    
      hlt_bitemu_hist_reco -> setBinLabel(j+1,hlt_bitnames[j].c_str());
      h_emu_reco -> setBinLabel(j+1,hlt_bitnames[j].c_str());
   
       }
    } 
  
      if (n_channel_==1 || n_channel_==2 || n_channel_==4){
   for (int j=0;j<n_hlt_bits_mu;j++) { 
     std::string histnameptmu  = "Muon1Pt_"+hlt_bitnamesMu[j];
     std::string histnameetamu = "Muon1Eta_"+hlt_bitnamesMu[j];
     h_ptmu1_trig[j] = dbe->book1D((histnameptmu).c_str(),(hlt_bitnamesMu[j]+"ptmuon").c_str(),50,0.0,150.0); 
     h_etamu1_trig[j] = dbe->book1D((histnameetamu).c_str(),(hlt_bitnamesMu[j]+"etamuon").c_str(),50,-2.5,2.5); 
     hlt_bitmu_hist_reco -> setBinLabel(j+1,hlt_bitnamesMu[j].c_str());
     h_mu_reco -> setBinLabel(j+1,hlt_bitnamesMu[j].c_str());
   
  }
   for (int j=0;j<n_hlt_bits_eg;j++) { 
     std::string histnameptel = "Electron1Pt_"+hlt_bitnamesEg[j];
     std::string histnameetael = "Electron1Eta_"+hlt_bitnamesEg[j];
     h_ptel1_trig[j] = dbe->book1D((histnameptel).c_str(),(hlt_bitnamesEg[j]+"ptelectron").c_str(),50,0.0,150.0);
     h_etael1_trig[j] = dbe->book1D((histnameetael).c_str(),(hlt_bitnamesEg[j]+"etaelectron").c_str(),50,-2.5,2.5);  
   
     hlt_bitel_hist_reco -> setBinLabel(j+1,hlt_bitnamesEg[j].c_str());
     h_el_reco -> setBinLabel(j+1,hlt_bitnamesEg[j].c_str());
   
   
    }
  
  }
  
    if (n_channel_==3){
   for (int j=0;j<n_hlt_bits_ph;j++) { 
     std::string histnameptph = "Photon1Pt_"+hlt_bitnamesPh[j];
     std::string histnameetaph = "Photon1Eta_"+hlt_bitnamesPh[j];
     h_ptph1_trig[j] = dbe->book1D((histnameptph).c_str(),(hlt_bitnamesPh[j]+"ptphoton").c_str(),50,0.0,200);
     h_etaph1_trig[j] = dbe->book1D((histnameetaph).c_str(),(hlt_bitnamesPh[j]+"etaphoton").c_str(),50,-2.5,2.5);
  
      hlt_bitph_hist_reco -> setBinLabel(j+1,hlt_bitnamesPh[j].c_str());
      h_ph_reco -> setBinLabel(j+1,hlt_bitnamesPh[j].c_str());
   
  }
  
  }
  
    if (n_channel_==5){
  for (int j=0;j<n_hlt_bits_tau;j++) { 
   //  std::string histnamepttau = "Tau1Pt_"+hlt_bitnamesTau[j];
   //  std::string histnameetatau = "Tau1Eta_"+hlt_bitnamesTau[j];
   //  h_pttau1_trig[j] = dbe->book1D((histnamepttau).c_str(),(hlt_bitnamesTau[j]+"pttau").c_str(),50,0.0,300);
   //  h_etatau1_trig[j] = dbe->book1D((histnameetatau).c_str(),(hlt_bitnamesTau[j]+"etatau").c_str(),50,-5.0,5.0);
      hlt_bittau_hist_gen -> setBinLabel(j+1,hlt_bitnamesTau[j].c_str());
      h_tau_gen -> setBinLabel(j+1,hlt_bitnamesTau[j].c_str());
    }
  }
  
 
 // std::cout << "booking OK " << std::endl;

}

HLTHiggsBits::~HLTHiggsBits()
{ }

//
// member functions
//


// ------------ method called to produce the data  ------------
void
HLTHiggsBits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // accumulation of statistics for HLT bits used by Higgs analysis 

  using namespace std;
  using namespace edm;


 
  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByLabel("muons", muonHandle);
  
  
  edm::Handle<reco::GsfElectronCollection> electronHandle;
  iEvent.getByLabel("gsfElectrons",electronHandle);
  
  edm::Handle<reco::PhotonCollection> photonHandle;
  iEvent.getByLabel("photons", photonHandle);
  
  edm::Handle<reco::CaloMETCollection> caloMet;
 // iEvent.getByLabel("met", caloMet);  // first attempt of adding met variables
 iEvent.getByLabel("corMetGlobalMuons", caloMet);

  edm::Handle<reco::TrackCollection> Tracks;
  iEvent.getByLabel("generalTracks", Tracks);
  
// MC truth part

  std::string errMsg("");
  edm::Handle<reco::CandidateView> mctruth;
  
 
  try {iEvent.getByLabel(mctruth_,mctruth);} catch (...) { errMsg=errMsg + "  -- No Gen Particles";}

  // do the MC-preselection. This depends on the channel under study. with
  // wrong n_channel the result would be nonsense
  if (n_channel_== 1) {
    mct_analysis_.analyzeHZZ4l(*mctruth, *muonHandle, *electronHandle, HltTree);
  } else if (n_channel_ == 2) {
  //  mct_analysis_.analyzeHWW2l(*mctruth, *muonHandle,*electronHandle, HltTree);
   mct_analysis_.analyzeHWW2l(*mctruth, *caloMet, *Tracks, *muonHandle,*electronHandle, HltTree);
  } else if (n_channel_ == 3) {
    mct_analysis_.analyzeHgg(*mctruth, *photonHandle, HltTree);
  } else if (n_channel_ == 4) {
    mct_analysis_.analyzeH2tau(*mctruth, HltTree);
  } else if (n_channel_ == 5) {
    mct_analysis_.analyzeHtaunu(*mctruth, HltTree);
  } else if (n_channel_ == 6) {
    mct_analysis_.analyzeHinv(*mctruth, HltTree);
  }



// HLT part

  // get hold of HL TriggerResults
  try {iEvent.getByLabel(hlTriggerResults_,HLTR);} catch (...) {;}
  if (!HLTR.isValid()) {
    LogDebug("") << "HL TriggerResults with label ["+hlTriggerResults_.encode()+"] not found!";
    return;
  }

  // initialisation
  if (!init_) {
    init_=true;
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*HLTR);
    hlNames_=triggerNames.triggerNames();
  }

 
  // define reco objects
  reco::Muon muon1, muon2;
  reco::GsfElectron electron1, electron2;
  reco::Photon photon1, photon2;
  
  
  //------------------------
  //  fill pt, eta reco objects
  //
  //---------------------------------
  
  
///////////////////////
///////7  Events passing reco muon preselection
/////////////////  
  if (mct_analysis_.MuonChannel_recoacc()) {  
    // trg_eff_gen_mc_mu -> Fill(4,1);
    
    if (n_channel_==2) h_met_hwwdimu->Fill(mct_analysis_.met_hwwdimu());
    
    if (n_channel_==4){
	  h_ptmu1->Fill(mct_analysis_.ptMuon1());
	  h_etamu1->Fill(mct_analysis_.etaMuon1());
	
	}
	else{  
    
         muon1 = mct_analysis_.muon1_();
         muon2 = mct_analysis_.muon2_();
 
           h_ptmu1->Fill(muon1.pt());
           h_ptmu2->Fill(muon2.pt());  
           h_etamu1->Fill(muon1.eta());
           h_etamu2->Fill(muon2.eta()); 
	   } 
	   
	 
	               
     }
  
/////////////////
////////  Events passing reco electron preselection
//////////////7  
  
   if (mct_analysis_.ElecChannel_recoacc()) {  
   //  trg_eff_gen_mc_elec -> Fill(4,1);
   
    if (n_channel_==2) h_met_hwwdiel->Fill(mct_analysis_.met_hwwdiel());
   
      if (n_channel_==4){
	  h_ptel1->Fill(mct_analysis_.ptElectron1());
	  h_etael1->Fill(mct_analysis_.etaElectron1());
	
	}
	else{
         electron1 = mct_analysis_.electron1_();
         electron2 = mct_analysis_.electron2_();
    
    
   //  std::cout<<"iso="<< electron1.dr03TkSumPt()<<std::endl;
    
        h_ptel1->Fill(electron1.pt());
        h_ptel2->Fill(electron2.pt());
        h_etael1->Fill(electron1.eta());
        h_etael2->Fill(electron2.eta());
	}
	
     }
  
////////////////
////////7  Events passing reco emu preselection
//////////  

   if (mct_analysis_.ElecMuChannel_recoacc()) {  
     //  trg_eff_gen_mc_emu -> Fill(4,1);
     
      if (n_channel_==2) h_met_hwwemu->Fill(mct_analysis_.met_hwwemu());
     
        if (n_channel_!=4){
        muon1 = mct_analysis_.muon1_();
        electron1 = mct_analysis_.electron1_();
    
        h_ptmu1_emu->Fill(muon1.pt());
        h_ptel1_emu->Fill(electron1.pt());
        h_etamu1_emu->Fill(muon1.eta());
        h_etael1_emu->Fill(electron1.eta());
	}
  
     }
  
 /////////////////
 /////  Events passing reco photon preselection
 ////////////7///////
  
   if (mct_analysis_.PhotonChannel_acc()) {  
 
 
      photon1 = mct_analysis_.photon1_();
      photon2 = mct_analysis_.photon2_();
    
     h_ptph1->Fill(photon1.pt());
     h_ptph2->Fill(photon2.pt());
     h_etaph1->Fill(photon1.eta());
     h_etaph2->Fill(photon2.eta());
    
  }
  
  
  /*   if (mct_analysis_.TauChannel_acc()) {  
     h_pttau1->Fill(mct_analysis_.ptTau1());
     h_etatau1->Fill(mct_analysis_.etaTau1());
 
     }
  */
  
  ///------------
  
  
  // decision for each HL algorithm
  const unsigned int n(hlNames_.size());
  
  // wtrig if set to 1 for paths that have fired
  int wtrig_m[100]={0};
  int wtrig_eg[100]={0};
  int wtrig_ph[100]={0};
  int wtrig_tau[100]={0};
  int wtrig_[100]={0};
  
  for (unsigned int i=0; i!=n; ++i) {
    if (HLTR->accept(i)) {
      for (int j=0;j<n_hlt_bits_mu;j++) {
        if (hlNames_[i] == hlt_bitnamesMu[j]) {     
	  wtrig_m[j]=1;
        }
      }
      for(int jj=0;jj<n_hlt_bits_eg;jj++) {
        if (hlNames_[i] == hlt_bitnamesEg[jj]) {     
	  wtrig_eg[jj]=1;
        }
      }
      
       for (int j=0;j<n_hlt_bits;j++) {
        if (hlNames_[i] == hlt_bitnames[j]) {     
	  wtrig_[j]=1;
        }
      }
      
      
       for(int k=0;k<n_hlt_bits_ph;k++) {
        if (hlNames_[i] == hlt_bitnamesPh[k]) {     
	  wtrig_ph[k]=1;
        }
      }
      for(int k=0;k<n_hlt_bits_tau;k++) {
        if (hlNames_[i] == hlt_bitnamesTau[k]) {     
	  wtrig_tau[k]=1;
        }
      }
      
      
    }
  }
  
 
  
  //// histos for muon, electron or photon paths
 
 //------------------------------------ 
 //        muons
 //-------------------------------------
   
  
   
    if (mct_analysis_.MuonChannel_recoacc()){
    
       for (int j=0;j<n_hlt_bits_mu;j++) {
          h_mu_reco->Fill(j+1);
       
          if (wtrig_m[j]==1) {  
	     hlt_bitmu_hist_reco->Fill(j+1); 
	  
	  if (n_channel_==4){
	    h_ptmu1_trig[j]->Fill(mct_analysis_.ptMuon1());
	   h_etamu1_trig[j]->Fill(mct_analysis_.etaMuon1());
	  
	  }
	  else{
	  
           h_ptmu1_trig[j]->Fill(muon1.pt());
	   h_etamu1_trig[j]->Fill(muon1.eta());
	   
	   }
          // hlt_bitmu_hist_reco->Fill(j+1);
       //h_ptmu2_trig[j]->Fill(mct_analysis_.ptmuon2());
          }
        }
      }
      
      
      
 //_------------------------------------
 //  electrons
 //_-----------------------------------
      
    
	
	  if (mct_analysis_.ElecChannel_recoacc()){
	  
	     
	     for (int j=0;j<n_hlt_bits_eg;j++) {
	         h_el_reco->Fill(j+1);
                 if (wtrig_eg[j]==1) {
		  hlt_bitel_hist_reco->Fill(j+1);
		 
		  if (n_channel_==4){
	    h_ptel1_trig[j]->Fill(mct_analysis_.ptElectron1());
	   h_etael1_trig[j]->Fill(mct_analysis_.etaElectron1());
	  
	  }
		 else {
                  h_ptel1_trig[j]->Fill(electron1.pt());
		  h_etael1_trig[j]->Fill(electron1.eta());
                //  hlt_bitel_hist_reco->Fill(j+1); 
                  }              
              }
	      }
        }
	
	
	
//-------------------------------------------------
//   emu channel
//
//----------------------------------------------------	
	
	
	 if (mct_analysis_.ElecMuChannel_recoacc()){
	   
	     for (int j=0;j<n_hlt_bits;j++) {
	         h_emu_reco->Fill(j+1);
                 if (wtrig_[j]==1) {
		 hlt_bitemu_hist_reco->Fill(j+1); 
		 
		   if (n_channel_!=4){
                  h_ptel1_emu_trig[j]->Fill(electron1.pt());
		  h_etael1_emu_trig[j]->Fill(electron1.eta());
		  h_ptmu1_emu_trig[j]->Fill(muon1.pt());
		  h_etamu1_emu_trig[j]->Fill(muon1.eta());
		  }
               //   hlt_bitemu_hist_reco->Fill(j+1); 
                  }              
              }
        }
	


//--------------------------------
//
//
//------------------------------	
    
    //photons reco
      if (mct_analysis_.PhotonChannel_acc()){
              //  h_ph_reco->Fill(1);
	  for (int j=0;j<n_hlt_bits_ph;j++) {
	      h_ph_reco->Fill(j+1);
             if (wtrig_ph[j]==1) {  
              h_ptph1_trig[j]->Fill(photon1.pt());
	      h_etaph1_trig[j]->Fill(photon1.eta());
              hlt_bitph_hist_reco->Fill(j+1);
            }                
          }
      }
      
      //taus
      if (mct_analysis_.TauChannel_acc()){
               //  h_tau->Fill(1);
		 
		// ev_clasif->Fill(mct_analysis_.evtype());
	  for (int j=0;j<n_hlt_bits_tau;j++) {
	      h_tau_gen->Fill(j+1);
             if (wtrig_tau[j]==1) {  
           //   h_pttau1_trig[j]->Fill(mct_analysis_.ptTau1());
	    //  h_etatau1_trig[j]->Fill(mct_analysis_.etaTau1());
              hlt_bittau_hist_gen->Fill(j+1);
            }                
          }
      }
      
  
  ////------------
 
    //----------------
    
    
  
  neventcount=nEvents_;

  return;

}


/*void
HLTHiggsBits::getL1Names(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
  iEvent.getByLabel(l1GTObjectMapTag_.label(), gtObjectMapRecord);

  const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
       gtObjectMapRecord->gtObjectMap();

  for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
       itMap != objMapVec.end(); ++itMap) {
    int algoBit = (*itMap).algoBitNumber();
    std::string algoNameStr = (*itMap).algoName();
    algoBitToName[algoBit] = algoNameStr;
  }
}*/




void
HLTHiggsBits::endJob()
{
  // final printout of accumulated statistics

 // std::cout << "Job ending " << std::endl;

  using namespace std;

 // std::cout << "Number of events handled:                      " << nEvents_ << std::endl;
 // std::cout << "Number of events seen in MC:                   " << n_inmc_ << ", (" << 100.0*n_inmc_/nEvents_ <<"%)" << std::endl;
 
 

//  return;



 // HltTree->Fill();
 // m_file->cd(); 
 // HltTree->Write();
 // delete HltTree;
  
 if(outputMEsInRootFile){
    dbe->showDirStructure();
    dbe->save(outputFileName);
  }

  
 // HltTree = 0;

//  if (m_file!=0) { // if there was a tree file...
 //   m_file->Write(); // write out the branches
 //   delete m_file; // close and delete the file
 //   m_file=0; // set to zero to clean up
 // }


  return;
}


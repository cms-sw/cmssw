#ifndef HLTHiggsBits_h
#define HLTHiggsBits_h

/** \class HLTHiggsBits
 *
 *  
 *  This class is an EDAnalyzer implementing TrigReport (statistics
 *  printed to log file) for HL triggers
 *
 *  $Date: 2010/02/16 22:33:12 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TNamed.h"
#include "TROOT.h"
#include "TChain.h"
#include <iostream>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
//#include "Geometry/Records/interface/IdealGeometryRecord.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


//Include DQM core
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"




#include<vector>
#include<string>

#include "HLTriggerOffline/Higgs/interface/HLTHiggsTruth.h"

//
// class declaration
//

class HLTHiggsBits : public edm::EDAnalyzer {

   public:
      explicit HLTHiggsBits(const edm::ParameterSet&);
      ~HLTHiggsBits();
      virtual void endJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

     
  




 edm::ParameterSet parameters;

 DQMStore* dbe;

     
      TTree *HltTree;
   /*   MonitorElement *hlt_mult_hist;
      MonitorElement *hlt_mult_hist_mc;
      MonitorElement *hlt_mult_hist_l1;
      MonitorElement *hlt_bit_hist;
      MonitorElement *hlt_bit_hist_reco;
      MonitorElement *hlt_bit_hist_mc;
      MonitorElement *hlt_bit_hist_l1;
      MonitorElement *hlt_bit_cumul;
      MonitorElement *hlt_bit_cumul_mc;
      MonitorElement *hlt_bit_cumul_l1;
      MonitorElement *hlt_redundancy[10];
      MonitorElement *hlt_redundancy_mc[10];
      MonitorElement *hlt_redundancy_l1[10];
      MonitorElement *trg_eff_gen;
      MonitorElement *trg_eff_gen_mc;
      
      
      MonitorElement *trg_eff_gen_mc_mu;
      MonitorElement *trg_eff_gen_mc_elec;
      MonitorElement *trg_eff_gen_mc_emu;
      
      
      
       MonitorElement *hlt_bitmu_hist;
       MonitorElement *hlt_bitel_hist;
       MonitorElement *hlt_bitemu_hist;
       MonitorElement *hlt_bitph_hist;
       MonitorElement *hlt_bittau_hist;
       
       
       MonitorElement *hlt_bitmu_hist_reco;
       MonitorElement *hlt_bitel_hist_reco;
       MonitorElement *hlt_bitemu_hist_reco;
       MonitorElement *hlt_bitph_hist_reco;
       MonitorElement *hlt_bittau_hist_reco;
        
	
	MonitorElement *ev_clasif;
       
       
	MonitorElement *h_mu;
	MonitorElement *h_el;
	MonitorElement *h_ph;
	MonitorElement *h_tau;
	MonitorElement *h_emu;
	MonitorElement *h_tot;
	MonitorElement *h_mc;
	MonitorElement *h_mc_reco;
	
	MonitorElement *h_mu_reco;
	MonitorElement *h_el_reco;
	MonitorElement *h_ph_reco;
	MonitorElement *h_tau_reco;
	MonitorElement *h_emu_reco;
	
	MonitorElement *h_l3pt;
	MonitorElement *h_l3eta;*/
      
       MonitorElement *h_ptmu1;
       MonitorElement *h_ptmu2;
       MonitorElement *h_ptmu1_trig[20];
       MonitorElement *h_ptmu2_trig[20];
       
       MonitorElement *h_ptel1;
       MonitorElement *h_ptel2;
       MonitorElement *h_ptel1_trig[20];
       MonitorElement *h_ptel2_trig[20];
       
       MonitorElement *h_ptmu1_emu;
       MonitorElement *h_etamu1_emu;
       MonitorElement *h_ptel1_emu;
       MonitorElement *h_etael1_emu;
       MonitorElement *h_ptmu1_emu_trig[20];
       MonitorElement *h_etamu1_emu_trig[20];
       MonitorElement *h_ptel1_emu_trig[20];
       MonitorElement *h_etael1_emu_trig[20];
       
       
       MonitorElement *h_ptph1;
       MonitorElement *h_ptph2;
       MonitorElement *h_ptph1_trig[20];
       MonitorElement *h_ptph2_trig[20];
       
    /*   MonitorElement *h_ptmu1_match[20];
       MonitorElement *h_ptmu2_match[20];
       
       MonitorElement *h_etamu1_match[20];
       MonitorElement *h_etamu2_match[20];
       
       MonitorElement *h_ptel1_match[20];
       MonitorElement *h_ptel2_match[20];
       
       MonitorElement *h_ptph1_match[20];
       MonitorElement *h_ptph2_match[20];*/
       
       MonitorElement *h_pttau1;
     //  MonitorElement *h_pttau2;
       MonitorElement *h_pttau1_trig[20];
      // MonitorElement *h_pttau2_trig[20];
      
       MonitorElement *h_etamu1;
       MonitorElement *h_etamu2;
       MonitorElement *h_etamu1_trig[20];
       MonitorElement *h_etamu2_trig[20];
       
       MonitorElement *h_etael1;
       MonitorElement *h_etael2;
       MonitorElement *h_etael1_trig[20];
       MonitorElement *h_etael2_trig[20];
       
       MonitorElement *h_etaph1;
       MonitorElement *h_etaph2;
       MonitorElement *h_etaph1_trig[20];
       MonitorElement *h_etaph2_trig[20];
       
       MonitorElement *h_etatau1;
     //  MonitorElement *h_etatau2;
       MonitorElement *h_etatau1_trig[20];
      // MonitorElement *h_etatau2_trig[20];
      
      
        MonitorElement* hlt_bitmu_hist_reco ;
        MonitorElement* h_mu_reco;
	
	MonitorElement* hlt_bitel_hist_reco ;
        MonitorElement* h_el_reco;
	
	MonitorElement* hlt_bitemu_hist_reco ;
        MonitorElement* h_emu_reco;
	
	MonitorElement* hlt_bitph_hist_reco;
	MonitorElement* h_ph_reco;
	
	MonitorElement* hlt_bittau_hist_gen;
	MonitorElement* h_tau_gen;

      
      
      MonitorElement *h_met_hwwdimu;
      MonitorElement *h_met_hwwdiel;
      MonitorElement *h_met_hwwemu;
      
      
     /*  MonitorElement *h_gen;
       MonitorElement *h_reco;
      
       MonitorElement *h_ptph1_l3;
       MonitorElement *h_etaph1_l3;
       MonitorElement *h_ptph2_l3;
       MonitorElement *h_etaph2_l3;*/
       

   private:

      edm::InputTag hlTriggerResults_;  // Input tag for TriggerResults
     /* edm::InputTag l1ParticleMapTag_;  // Input tag for L1-Particles maps
      edm::InputTag l1GTReadoutRecTag_; // Input tag for Tag for L1 Trigger Results
      edm::InputTag l1GTObjectMapTag_;*/
      edm::InputTag mctruth_;          
           
      int n_channel_;
      int n_hlt_bits, n_hlt_bits_eg, n_hlt_bits_mu, n_hlt_bits_ph, n_hlt_bits_tau;
      
    //  std::map<int, std::string> algoBitToName;

      HLTHiggsTruth mct_analysis_;

      edm::Handle<edm::TriggerResults> HLTR;
      
    /*

      unsigned int  n_true_;              // total selected (taken by HLT, passing MC preselection)
      unsigned int  n_fake_;              // total fakes (taken by HLT, not passing MC preselection)
      unsigned int  n_miss_;              // total misses (passing MC preselection, not taken by HLT)
      unsigned int  n_inmc_;              // total of events passing MC preselection
      unsigned int  n_L1acc_;	          // total of L1 accepts
      unsigned int  n_L1acc_mc_;          // L1 accepts of events, passing MC preselection
      unsigned int  n_hlt_of_L1_;         // HLT accepts of all L1 accepted
      unsigned int n_mm_;
      unsigned int n_mm_acc;
      unsigned int n_ee_;
      unsigned int n_ee_acc;
      unsigned int n_em_;*/
      
      unsigned int  nEvents_;           // number of events processed
      std::vector<std::string>  hlNames_;  // name of each HLT algorithm
      bool init_;                          // vectors initialised or not

      bool l1_decision;

      int hlt_whichbit[3][20][21];
     

      TFile* m_file; // pointer to Histogram file
      int *hlt_nbits;
      int neventcount;

      std::string histName;
      std::vector<std::string> hlt_bitnames;
      std::vector<std::string> hlt_bitnamesMu;
      std::vector<std::string> hlt_bitnamesEg;
   
      std::vector<std::string> hlt_bitnamesPh;
      std::vector<std::string> hlt_bitnamesTau;
      
    /*  std::vector<std::string> hlt_labels;
      std::vector<std::string> hlt_labelsMu;
      std::vector<std::string> hlt_labelsEg;*/


      std::ofstream outfile;
      
    /*  typedef math::XYZTLorentzVector LorentzVector;*/


    
    std::string triggerTag_;
    std::string outFile_, outputFileName;
    bool outputMEsInRootFile;
    
      


};

#endif //HLTHiggsBits_h

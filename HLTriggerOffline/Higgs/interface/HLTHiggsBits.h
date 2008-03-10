#ifndef HLTHiggsBits_h
#define HLTHiggsBits_h

/** \class HLTHiggsBits
 *
 *  
 *  This class is an EDAnalyzer implementing TrigReport (statistics
 *  printed to log file) for HL triggers
 *
 *  $Date: 2007/06/19 11:47:50 $
 *  $Revision: 1.2 $
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
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
      virtual void getL1Names(const edm::Event& iEvent, const edm::EventSetup& iSetup);
      virtual void analyze(const edm::Event&, const edm::EventSetup&);

      TTree *HltTree;
      TH1D *hlt_mult_hist;
      TH1D *hlt_mult_hist_mc;
      TH1D *hlt_mult_hist_l1;
      TH1D *hlt_bit_hist;
      TH1D *hlt_bit_hist_mc;
      TH1D *hlt_bit_hist_l1;
      TH1D *hlt_bit_cumul;
      TH1D *hlt_bit_cumul_mc;
      TH1D *hlt_bit_cumul_l1;
      TH1D *hlt_redundancy[10];
      TH1D *hlt_redundancy_mc[10];
      TH1D *hlt_redundancy_l1[10];
      TH1D *trg_eff_gen;
      TH1D *trg_eff_gen_mc;

   private:

      edm::InputTag hlTriggerResults_;  // Input tag for TriggerResults
      edm::InputTag l1ParticleMapTag_;  // Input tag for L1-Particles maps
      edm::InputTag l1GTReadoutRecTag_; // Input tag for Tag for L1 Trigger Results
      edm::InputTag l1GTObjectMapTag_;
      edm::InputTag mctruth_;           // Input tag for Tag for L1 Trigger Results
      int n_channel_, n_hlt_bits;
      edm::TriggerNames triggerNames_;  // TriggerNames class

      std::map<int, std::string> algoBitToName;

      HLTHiggsTruth mct_analysis_;

      edm::Handle<edm::TriggerResults> HLTR;

      unsigned int  n_true_;              // total selected (taken by HLT, passing MC preselection)
      unsigned int  n_fake_;              // total fakes (taken by HLT, not passing MC preselection)
      unsigned int  n_miss_;              // total misses (passing MC preselection, not taken by HLT)
      unsigned int  n_inmc_;              // total of events passing MC preselection
      unsigned int  n_L1acc_;	          // total of L1 accepts
      unsigned int  n_L1acc_mc_;          // L1 accepts of events, passing MC preselection
      unsigned int  n_hlt_of_L1_;         // HLT accepts of all L1 accepted

      unsigned int  nEvents_;           // number of events processed
      std::vector<std::string>  hlNames_;  // name of each HLT algorithm
      bool init_;                          // vectors initialised or not

      bool l1_decision;

      int hlt_whichbit[3][20][21];


      TFile* m_file; // pointer to Histogram file
      int *hlt_nbits;
      int neventcount;

      string histName;
      std::vector<string> hlt_bitnames;

      ofstream outfile;

};

#endif //HLTHiggsBits_h

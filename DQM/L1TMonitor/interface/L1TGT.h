#ifndef L1TGT_H
#define L1TGT_H

/*
 * \file L1TGT.h
 *
 * $Date: 2008/04/25 14:57:19 $
 * $Revision: 1.6 $
 * \author J. Berryhill, I. Mikulec
 *
*/

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TGT : public edm::EDAnalyzer {

public:

// Constructor
L1TGT(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TGT();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob();

// EndJob
void endJob(void);

private:
  
  bool isActive(int word, int bit);
  // Active boards DAQ record bit number:
  // 0 FDL 
  // 1 PSB_0 9 Techn.Triggers for FDL
  // 2 PSB_1 13 Calo data for GTL
  // 3 PSB_2 14 Calo data for GTL
  // 4 PSB_3 15 Calo data for GTL
  // 5 PSB_4 19 M/Q bits for GMT
  // 6 PSB_5 20 M/Q bits for GMT
  // 7 PSB_6 21 M/Q bits for GMT
  // 8 GMT
  enum activeDAQ { FDL=0, PSB9, PSB13, PSB14, PSB15, PSB19, PSB20, PSB21, GMT };
  // Active boards EVM record bit number:
  // 0 TCS 
  // 1 FDL 
  enum activeEVM { TCS, FDLEVM };
  
  // ----------member data ---------------------------
  DQMStore * dbe;

  MonitorElement* algo_bits;
  MonitorElement* algo_bits_corr;
  MonitorElement* tt_bits;
  MonitorElement* tt_bits_corr;
  MonitorElement* algo_tt_bits_corr;
  MonitorElement* algo_bits_lumi;
  MonitorElement* tt_bits_lumi;
  MonitorElement* event_type;
 
  MonitorElement* event_number;
  MonitorElement* event_lumi;
  MonitorElement* trigger_number;
  MonitorElement* trigger_lumi;
  MonitorElement* evnum_trignum_lumi;
  MonitorElement* orbit_lumi;
  MonitorElement* setupversion_lumi; 

  MonitorElement* gtfe_bx;
  MonitorElement* dbx_module;

  MonitorElement* BST_MasterStatus;
  MonitorElement* BST_turnCountNumber;
  MonitorElement* BST_lhcFillNumber;
  MonitorElement* BST_beamMode;
  MonitorElement* BST_beamMomentum;
  MonitorElement* BST_intensityBeam1;
  MonitorElement* BST_intensityBeam2;
  MonitorElement* gpsfreq;
  MonitorElement* gpsfreqwide;
  MonitorElement* gpsfreqlum;
  
  

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag gtSource_;
  edm::InputTag gtEvmSource_;
  
  boost::uint64_t preGps_;
  boost::uint64_t preOrb_;
};

#endif

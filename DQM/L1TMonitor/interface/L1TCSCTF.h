#ifndef L1TCSCTF_H
#define L1TCSCTF_H

/*
 * \file L1TCSCTF.h
 *
 * $Date: 2007/07/19 16:48:23 $
 * $Revision: 1.3 $
 * \author J. Berryhill
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

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TCSCTF : public edm::EDAnalyzer {

public:

// Constructor
L1TCSCTF(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TCSCTF();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DaqMonitorBEInterface * dbe;

  MonitorElement* csctfetavalue;
  MonitorElement* csctfphivalue;
  MonitorElement* csctfptvalue;
  MonitorElement* csctfptpacked;
  MonitorElement* csctfquality;
  MonitorElement* csctfchargevalue;
  MonitorElement* csctfntrack;

  // Below are two sets of MonitorElement:
  //  first is modified from what initially was existing in this class
  //  second is what we also want to see to monitor our system

  // If you monitor CSC TF hardware output data, only "packed" data are available
  MonitorElement* csctfetapacked[3];
  MonitorElement* csctfphipacked[3];
//  MonitorElement* csctfptpacked[3];
  MonitorElement* csctfchargepacked[3];
//  MonitorElement* csctfquality[3];

  MonitorElement* csctfbx;
  // We want to monitor our input data too
  MonitorElement* csctfnlcts;
  MonitorElement* csctflctbx;
  MonitorElement* csctflctquality;
//  MonitorElement* csctflctxy;



///KK
  // Type of input data for the DQM
  bool emulation;

  // geometry may not be properly set in CSC TF hardware (and in data respectively)
  // make an artificial assignment of each of 12 SPs (slots 6-11 and 16-21) to 12 sectors (1-12, 0-not assigned)
  std::vector<int> slot2sector;

  // following arrays are indexed by sector # (1-12)
  //   and have one spare element [0] for unknown SP board ID (in case of corrupted data)
  MonitorElement* cscsp_fmm_status[13]; // FMM status for each SP
  MonitorElement* cscsp_errors[13];     // Logical 'OR' of various data errors that SP can detect
  MonitorElement* csctf_errors;         // Cumulative errors for the whole TF crate
///
  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag csctfSource_;



};

#endif

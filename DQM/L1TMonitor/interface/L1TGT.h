#ifndef L1TGT_H
#define L1TGT_H

/*
 * \file L1TGT.h
 *
 * $Date: 2007/04/03 20:04:01 $
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

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

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
void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DaqMonitorBEInterface * dbe;

  MonitorElement* gttriggerdword;
  MonitorElement* gttriggerdbits;
  MonitorElement* gttriggerdbitscorr;
  MonitorElement* gtfdlbx;
  MonitorElement* gtfdlevent;
  MonitorElement* gtfdllocalbx;
  MonitorElement* gtfdlbxinevent;
  MonitorElement* gtfdlsize;

  MonitorElement* gtfeboardId;
  MonitorElement* gtferecordlength;
  MonitorElement* gtfebx;
  MonitorElement* gtfesetupversion; 
  MonitorElement* gtfeactiveboards; 
  MonitorElement* gtfetotaltrigger;
  MonitorElement* gtfesize;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag gtSource_;
};

#endif

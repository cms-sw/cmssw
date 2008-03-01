#ifndef L1TCSCTF_H
#define L1TCSCTF_H

/*
 * \file L1TCSCTF.h
 *
 * $Date: 2007/12/03 14:24:08 $
 * $Revision: 1.6 $
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

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
  DQMStore * dbe;

  MonitorElement* csctfetavalue[3];
  MonitorElement* csctfphivalue[3];
  MonitorElement* csctfptvalue[3];
  MonitorElement* csctfchargevalue[3];
  MonitorElement* csctfquality[3];
  MonitorElement* csctfntrack;
  MonitorElement* csctfbx;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag csctfSource_ ;
};

#endif

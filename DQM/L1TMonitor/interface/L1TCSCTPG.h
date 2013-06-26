#ifndef L1TCSCTPG_H
#define L1TCSCTPG_H

/*
 * \file L1TCSCTPG.h
 *
 * $Date: 2009/11/19 14:30:34 $
 * $Revision: 1.4 $
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

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TCSCTPG : public edm::EDAnalyzer {

public:

// Constructor
L1TCSCTPG(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TCSCTPG();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
void beginJob(void);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  MonitorElement* csctpgpattern;
  MonitorElement* csctpgquality;
  MonitorElement* csctpgwg;
  MonitorElement* csctpgstrip;
  MonitorElement* csctpgstriptype;
  MonitorElement* csctpgbend;
  MonitorElement* csctpgbx;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag csctpgSource_;
};

#endif

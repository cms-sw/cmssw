#ifndef L1TCSCTF_H
#define L1TCSCTF_H

/*
 * \file L1TCSCTF.h
 *
 * $Date: 2008/03/01 00:40:00 $
 * $Revision: 1.7 $
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

// KK_start: Sector Receiver LUT class to transform wire/strip numbers to eta/phi observables
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
// KK_end

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

  // KK_start: see source for description
  MonitorElement* csctferrors;
  MonitorElement* csctfoccupancies;
  CSCSectorReceiverLUT *srLUTs_[5];
  // KK_end

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag gmtProducer, lctProducer, trackProducer, statusProducer;
};

#endif

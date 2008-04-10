// -*-C++-*- 
#ifndef L1TECALTPG_H
#define L1TECALTPG_H

/*
 * \file L1TECALTPG.h
 *
 * $Date: 2007/08/29 14:02:45 $
 * $Revision: 1.5 $
 * \author J. Berryhill
 *
 * $Log: L1TECALTPG.h,v $
 * Revision 1.5  2007/08/29 14:02:45  wittich
 * split into barrel and endcap
 *
 * Revision 1.4  2007/02/22 19:43:52  berryhil
 *
 *
 *
 * InputTag parameters added for all modules
 *
 * Revision 1.3  2007/02/20 22:48:59  wittich
 * - change from getByType to getByLabel in ECAL TPG,
 *   and make it configurable.
 * - fix problem in the GCT with incorrect labels. Not the ultimate
 *   solution - will probably have to go to many labels.
 *
 * Revision 1.2  2007/02/19 22:07:26  wittich
 * - Added three monitorables to the ECAL TPG monitoring (from GCTMonitor)
 * - other minor tweaks in GCT, etc
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
// class declaration
//

class L1TECALTPG:public edm::EDAnalyzer {

public:

  // Constructor
  L1TECALTPG(const edm::ParameterSet & ps);

  // Destructor
  virtual ~ L1TECALTPG();

protected:
  // Analyze
  void analyze(const edm::Event & e, const edm::EventSetup & c);

  // BeginJob
  void beginJob(const edm::EventSetup & c);

  // EndJob
  void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  // what we monitor
  MonitorElement *ecalTpEtEtaPhiB_;
  MonitorElement *ecalTpOccEtaPhiB_;
  MonitorElement *ecalTpRankB_;

  MonitorElement *ecalTpEtEtaPhiE_;
  MonitorElement *ecalTpOccEtaPhiE_;
  MonitorElement *ecalTpRankE_;

  int nev_;			// Number of events processed
  std::string outputFile_;	//file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;

  edm::InputTag ecaltpgSourceB_; // barrel
  edm::InputTag ecaltpgSourceE_; // endcap

};

#endif

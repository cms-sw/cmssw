// -*-C++-*- 
#ifndef L1TECALTPG_H
#define L1TECALTPG_H

/*
 * \file L1TECALTPG.h
 *
 * $Date: 2007/02/19 19:24:08 $
 * $Revision: 1.1 $
 * \author J. Berryhill
 *
 * $Log$
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
  DaqMonitorBEInterface * dbe;

  // what we monitor
  MonitorElement *ecalTpEtEtaPhi_;
  MonitorElement *ecalTpOccEtaPhi_;
  MonitorElement *ecalTpRank_;

  int nev_;			// Number of events processed
  std::string outputFile_;	//file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;

};

#endif

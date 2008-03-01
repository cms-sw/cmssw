#ifndef L1THCALTPG_H
#define L1THCALTPG_H

/*
 * \file L1THCALTPG.h
 *
 * $Date: 2007/02/23 22:00:16 $
 * $Revision: 1.3 $
 * \author J. Berryhill
 *
*/

// system include files
#include <memory>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>

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


#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveDigi.h"

//
// class decleration
//

class L1THCALTPG : public edm::EDAnalyzer {

public:

// Constructor
L1THCALTPG(const edm::ParameterSet& ps);

// Destructor
virtual ~L1THCALTPG();

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

  // what we monitor
  MonitorElement *hcalTpEtEtaPhi_;
  MonitorElement *hcalTpOccEtaPhi_;
  MonitorElement *hcalTpRank_;

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag hcaltpgSource_;
};

#endif

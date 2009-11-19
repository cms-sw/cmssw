#ifndef L1TLTC_H
#define L1TLTC_H

/*
 * \file L1TLTC.h
 *
 * $Date: 2008/03/01 00:40:00 $
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/LTCDigi/interface/LTCDigi.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TLTC : public edm::EDAnalyzer {

public:

// Constructor
L1TLTC(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TLTC();

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

  // ----------member data ---------------------------
  MonitorElement* h1;
  MonitorElement* h2;
  MonitorElement* h3;
  //MonitorElement* h4;
  MonitorElement* overlaps;
  MonitorElement* n_inhibit;
  MonitorElement* run;
  MonitorElement* event;
  MonitorElement* gps_time;
  float XMIN; float XMAX;
  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  ofstream logFile_;
  edm::InputTag ltcSource_ ;
};

#endif

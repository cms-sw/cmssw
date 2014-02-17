#ifndef L1TFED_H
#define L1TFED_H

/*
 * \file L1TFED.h
 *
 * $Date: 2010/04/06 01:14:01 $
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

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TFED : public edm::EDAnalyzer {

public:

// Constructor
L1TFED(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TFED();

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

//  MonitorElement* fedtest;
  MonitorElement * hfedsize;
  MonitorElement * hfedprof;
//  MonitorElement ** hindfed;
  
  MonitorElement* fedentries; 
  MonitorElement* fedfatal;
  MonitorElement* fednonfatal;  

  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::vector<int> l1feds_;
  ofstream logFile_;
  edm::InputTag fedSource_;  
  edm::InputTag rawl_;
  std::string directory_;
  bool stableROConfig_;
};

#endif

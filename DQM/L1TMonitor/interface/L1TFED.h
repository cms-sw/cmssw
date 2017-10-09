#ifndef L1TFED_H
#define L1TFED_H

/*
 * \file L1TFED.h
 *
 * \author J. Berryhill
 *
*/

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
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

class L1TFED : public DQMEDAnalyzer {

public:

// Constructor
L1TFED(const edm::ParameterSet& ps);

// Destructor
virtual ~L1TFED();

protected:
// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c) override;

// BeginRun
void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // ----------member data ---------------------------
  MonitorElement * hfedsize;
  MonitorElement * hfedprof;
  
  MonitorElement* fedentries; 
  MonitorElement* fedfatal;
  MonitorElement* fednonfatal;  

  int nev_; // Number of events processed
  bool verbose_;
  bool monitorDaemon_;
  std::vector<int> l1feds_;
  std::ofstream logFile_;
  edm::InputTag fedSource_;  
  edm::EDGetTokenT<FEDRawDataCollection> rawl_;
  std::string directory_;
  bool stableROConfig_;
};

#endif

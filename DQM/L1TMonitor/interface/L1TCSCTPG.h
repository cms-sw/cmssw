#ifndef L1TCSCTPG_H
#define L1TCSCTPG_H

/*
 * \file L1TCSCTPG.h
 *
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
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class L1TCSCTPG : public DQMEDAnalyzer {
public:
  // Constructor
  L1TCSCTPG(const edm::ParameterSet& ps);

  // Destructor
  ~L1TCSCTPG() override;

protected:
  // Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;

private:
  // ----------member data ---------------------------

  MonitorElement* csctpgpattern;
  MonitorElement* csctpgquality;
  MonitorElement* csctpgwg;
  MonitorElement* csctpgstrip;
  MonitorElement* csctpgstriptype;
  MonitorElement* csctpgbend;
  MonitorElement* csctpgbx;

  int nev_;                 // Number of events processed
  std::string outputFile_;  //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  edm::InputTag csctpgSource_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> csctpgSource_token_;
};

#endif

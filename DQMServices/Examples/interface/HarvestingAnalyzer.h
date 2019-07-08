#ifndef HarvestingAnalyzer_h
#define HarvestingAnalyzer_h

/** \class HarvestingAnalyzer
 *
 *  Class to perform operations on MEs after EDMtoMEConverter
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "TString.h"

class HarvestingAnalyzer : public edm::EDAnalyzer {
public:
  explicit HarvestingAnalyzer(const edm::ParameterSet &);
  ~HarvestingAnalyzer() override;
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;

private:
  std::string fName;
  int verbosity;
  DQMStore *dbe;
};

#endif

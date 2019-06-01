#ifndef HarvestingDataCertification_h
#define HarvestingDataCertification_h

/** \class HarvestingDataCertification
 *
 *  Class to fill dqm monitor elements from existing EDM file
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
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class HarvestingDataCertification : public edm::EDAnalyzer {
public:
  explicit HarvestingDataCertification(const edm::ParameterSet &);
  ~HarvestingDataCertification() override;
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;

private:
  std::string fName;
  int verbosity;
};

#endif

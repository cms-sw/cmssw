#ifndef ConverterTester_h
#define ConverterTester_h

/** \class ConverterTester
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
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "TRandom.h"
#include "TRandom3.h"
#include "TString.h"

class ConverterTester : public edm::EDAnalyzer {
public:
  explicit ConverterTester(const edm::ParameterSet &);
  ~ConverterTester() override;
  void beginJob() override;
  void endJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;

private:
  std::string fName;
  int verbosity;
  int frequency;
  std::string label;
  DQMStore *dbe;

  MonitorElement *meTestString;
  MonitorElement *meTestInt;
  MonitorElement *meTestFloat;
  MonitorElement *meTestTH1FD;
  MonitorElement *meTestTH1FN;
  MonitorElement *meTestTH2F;
  MonitorElement *meTestTH3F;
  MonitorElement *meTestProfile1;
  MonitorElement *meTestProfile2;

  TRandom *Random;
  double RandomVal1;
  double RandomVal2;
  double RandomVal3;

  // private statistics information
  unsigned int count;
};

#endif

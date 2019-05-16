#ifndef QualityTester_H
#define QualityTester_H

/*
 * \file QualityTester.h
 *
 * Helping EDAnalyzer running the quality tests for clients when:
 * - they receive ME data from the SM 
 * - they are run together with the producers (standalone mode)
 *
 * \author M. Zanetti - CERN PH
 *
 */
#include "boost/scoped_ptr.hpp"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/DQMXMLFileRcd.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <memory>
#include <iostream>
#include <string>

class QTestHandle;


class QualityTester: public edm::EDAnalyzer{

public:

  /// Constructor
  QualityTester(const edm::ParameterSet& ps);
  
  /// Destructor
  ~QualityTester() override;

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) override ;

  /// perform the actual quality tests
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c) override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override ;
  void endRun(const edm::Run& r, const edm::EventSetup& c) override;
  void endJob() override;

private:


  void performTests();
  
  int nEvents;
  int prescaleFactor;
  bool getQualityTestsFromFile;
  std::string Label;
  bool testInEventloop;
  bool qtestOnEndRun;
  bool qtestOnEndJob;
  bool qtestOnEndLumi;
  std::string reportThreshold;
  bool verboseQT;

  DQMStore * bei;

  QTestHandle * qtHandler;
};

#endif

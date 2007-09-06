#ifndef QualityTester_H
#define QualityTester_H

/*
 * \file QualityTester.h
 *
 * Helping EDAnalyzer running the quality tests for clients when:
 * - they receive ME data from the SM 
 * - they are run together with the producers (standalone mode)
 *
 * $Date: 2007/09/06 13:21:30 $
 * $Revision: 1.2 $
 * \author M. Zanetti - CERN PH
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <memory>
#include <iostream>
#include <string>

class MonitorUserInterface;
class DaqMonitorBEInterface;
class QTestHandle;


class QualityTester: public edm::EDAnalyzer{

public:

  /// Constructor
  QualityTester(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~QualityTester();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) {};

  /// perform the actual quality tests
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);


private:

  int prescaleFactor;
  bool getQualityTestsFromFile;

  edm::ParameterSet parameters;

  MonitorUserInterface * mui;
  DaqMonitorBEInterface * bei;

  QTestHandle * qtHandler;


};

#endif

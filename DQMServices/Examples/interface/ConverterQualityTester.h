#ifndef ConverterQualityTester_h
#define ConverterQualityTester_h

/** \class ConverterQualityTester
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2008/03/26 22:16:30 $
 *  $Revision: 1.1 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

class ConverterQualityTester : public edm::EDAnalyzer
{
  
 public:

  explicit ConverterQualityTester(const edm::ParameterSet&);
  virtual ~ConverterQualityTester();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  
private:
  std::string fName;
  int verbosity;
};

#endif

#ifndef ConverterTester_h
#define ConverterTester_h

/** \class GlobalHitsAnalyzer
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2008/03/13 21:17:07 $
 *  $Revision: 1.5 $
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

#include "TString.h"
#include "TRandom.h"
#include "TRandom3.h"

class ConverterTester : public edm::EDAnalyzer
{
  
 public:

  explicit ConverterTester(const edm::ParameterSet&);
  virtual ~ConverterTester();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  
private:
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  std::string label;
  DQMStore *dbe;
  std::string outputfile;
  bool doOutput;

  MonitorElement *meTestString;
  MonitorElement *meTestInt;
  MonitorElement *meTestFloat;
  MonitorElement *meTestTH1F;
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

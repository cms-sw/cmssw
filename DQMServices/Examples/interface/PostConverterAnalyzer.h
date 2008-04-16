#ifndef PostConverterAnalyzer_h
#define PostConverterAnalyzer_h

/** \class PostConverterAnalyzer
 *  
 *  Class to perform operations on MEs after EDMtoMEConverter
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

#include "TString.h"

class PostConverterAnalyzer : public edm::EDAnalyzer
{
  
 public:

  explicit PostConverterAnalyzer(const edm::ParameterSet&);
  virtual ~PostConverterAnalyzer();
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  
private:
  std::string fName;
  int verbosity;
  DQMStore *dbe;

};

#endif

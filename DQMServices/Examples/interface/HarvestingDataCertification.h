#ifndef HarvestingDataCertification_h
#define HarvestingDataCertification_h

/** \class HarvestingDataCertification
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2009/12/14 22:22:22 $
 *  $Revision: 1.2 $
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

class HarvestingDataCertification : public edm::EDAnalyzer
{
  
 public:

  explicit HarvestingDataCertification(const edm::ParameterSet&);
  virtual ~HarvestingDataCertification();
  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  
private:
  std::string fName;
  int verbosity;
};

#endif

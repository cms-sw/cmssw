#ifndef HiggsHarvesting_h
#define HiggsHarvesting_h

/** \class HiggsHarvesting
 *  
 *  Class to perform operations on MEs after EDMtoMEConverter
 *
 *  \author Hugues Brun
 */

// framework & common header files
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
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"

class HiggsHarvesting : public DQMEDHarvester
{
  
 public:

  explicit HiggsHarvesting(const edm::ParameterSet&);
  virtual ~HiggsHarvesting();
  virtual void beginJob();
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);

  
private:
  std::string analysisName;

};

#endif

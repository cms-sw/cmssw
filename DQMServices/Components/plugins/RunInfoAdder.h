#ifndef RunInfoAdder_H
#define RunInfoAdder_H

/** \class RunInfoAdder
 *
 * A Harvester plugin that adds information about the run (mainly the run
 * number) to every histogram, to prevent losing or confusing this information
 * when histograms are handled as graphics files.
 *
 *  \author Marcel Schneider (marcel.andre.schneider@cern.ch)
 */

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
 
class RunInfoAdder : public DQMEDHarvester{

public:
  RunInfoAdder(const edm::ParameterSet& ps);
  virtual ~RunInfoAdder();
  
protected:
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override;
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  //variables from config file
  bool addRunNumber_;
  bool addLumi_;
  std::vector<std::string> folder_;

  // information to add, has to be saved from endLumi
  uint32_t run_;
};

#endif

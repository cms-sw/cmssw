#ifndef DQMSERVICES_DATABASEIO_DQMDATABASEHARVESTER_H
#define DQMSERVICES_DATABASEIO_DQMDATABASEHARVESTER_H

// Framework
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMDatabaseWriter.h"

class DQMDatabaseHarvester : public DQMEDHarvester {
 public:
  DQMDatabaseHarvester(const edm::ParameterSet &ps);
  virtual ~DQMDatabaseHarvester();

 protected:
  void beginJob();
  virtual void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &,
                                     edm::LuminosityBlock const &,
                                     edm::EventSetup const &) override;
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;
  virtual void endRun(edm::Run const &, edm::EventSetup const &) override;

 private:
  // variables from config file

  std::string s_histogramsPath;
  std::vector<std::string> vs_histogramsPerLumi;
  std::vector<std::string> vs_histogramsPerRun;

  std::vector<MonitorElement *> histogramsPerLumi;
  std::vector<std::pair<MonitorElement *, HistogramValues> > histogramsPerRun;

  std::unique_ptr<DQMDatabaseWriter> dbw_;
};

#endif

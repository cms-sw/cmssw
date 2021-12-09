#ifndef DQM_L1TMONITORCLIENT_L1TdeStage2ShowerCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TdeStage2ShowerCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TdeStage2ShowerClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TdeStage2ShowerClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TdeStage2ShowerClient() override;

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &);

  std::string monitorDir_;

  MonitorElement *emtfShowerDataSummary_eff_;
  MonitorElement *emtfShowerEmulSummary_eff_;
};

#endif

#ifndef DQM_L1TMONITORCLIENT_L1TdeCSCTPGShowerCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TdeCSCTPGShowerCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <string>

class L1TdeCSCTPGShowerClient : public DQMEDHarvester {
public:
  /// Constructor
  L1TdeCSCTPGShowerClient(const edm::ParameterSet &ps);

  /// Destructor
  ~L1TdeCSCTPGShowerClient() override;

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

  MonitorElement *lctShowerDataSummary_eff_;
  MonitorElement *alctShowerDataSummary_eff_;
  MonitorElement *clctShowerDataSummary_eff_;
  MonitorElement *lctShowerEmulSummary_eff_;
  MonitorElement *alctShowerEmulSummary_eff_;
  MonitorElement *clctShowerEmulSummary_eff_;
};

#endif

#ifndef DQM_L1TMONITORCLIENT_L1TdeCSCTPGShowerCLIENT_H
#define DQM_L1TMONITORCLIENT_L1TdeCSCTPGShowerCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
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

  MonitorElement *lctShowerDataNomSummary_eff_;
  MonitorElement *alctShowerDataNomSummary_eff_;
  MonitorElement *clctShowerDataNomSummary_eff_;
  MonitorElement *lctShowerEmulNomSummary_eff_;
  MonitorElement *alctShowerEmulNomSummary_eff_;
  MonitorElement *clctShowerEmulNomSummary_eff_;

  MonitorElement *lctShowerDataTightSummary_eff_;
  MonitorElement *alctShowerDataTightSummary_eff_;
  MonitorElement *clctShowerDataTightSummary_eff_;
  MonitorElement *lctShowerEmulTightSummary_eff_;
  MonitorElement *alctShowerEmulTightSummary_eff_;
  MonitorElement *clctShowerEmulTightSummary_eff_;
};

#endif

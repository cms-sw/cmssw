#ifndef RPCMonitorClient_RPCDCSSummary_H
#define RPCMonitorClient_RPCDCSSummary_H

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include <map>

class RPCDCSSummary : public DQMEDHarvester {
public:
  RPCDCSSummary(const edm::ParameterSet &);
  ~RPCDCSSummary() override = default;

protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  void myBooker(DQMStore::IBooker &);
  void checkDCSbit(edm::EventSetup const &);

  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;

  bool init_;
  double defaultValue_;

  bool offlineDQM_;

  MonitorElement *DCSMap_;
  MonitorElement *totalDCSFraction;
  constexpr static int nWheels_ = 5;
  MonitorElement *dcsWheelFractions[nWheels_];
  constexpr static int nDisks_ = 10;
  MonitorElement *dcsDiskFractions[nDisks_];
  std::pair<int, int> FEDRange_;
  int numberOfDisks_;
  int NumberOfFeds_;
};

#endif

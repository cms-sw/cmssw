#ifndef DQM_RPCMonitorClient_DQMDaqInfo_H
#define DQM_RPCMonitorClient_DQMDaqInfo_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include <utility>

class RPCDaqInfo : public DQMEDHarvester {
public:
  explicit RPCDaqInfo(const edm::ParameterSet &);
  ~RPCDaqInfo() override = default;

protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;       //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;  //performed in the endJob

private:
  void myBooker(DQMStore::IBooker &);

  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;

  bool init_;

  MonitorElement *DaqFraction_;
  MonitorElement *DaqMap_;
  constexpr static int nWheels_ = 5;
  MonitorElement *daqWheelFractions[nWheels_];
  constexpr static int nDisks_ = 10;
  MonitorElement *daqDiskFractions[nDisks_];

  std::pair<int, int> FEDRange_;

  int numberOfDisks_, NumberOfFeds_;
};

#endif

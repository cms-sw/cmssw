#ifndef RPCMonitorClient_RPCDataCertification_H
#define RPCMonitorClient_RPCDataCertification_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

class RPCDataCertification : public DQMEDHarvester {
public:
  RPCDataCertification(const edm::ParameterSet& pset);
  ~RPCDataCertification() override = default;

protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker&,
                             DQMStore::IGetter&,
                             edm::LuminosityBlock const&,
                             edm::EventSetup const&) override;      //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;  //performed in the endJob

private:
  void myBooker(DQMStore::IBooker&);
  void checkFED(edm::EventSetup const&);

  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;

  MonitorElement* CertMap_;
  MonitorElement* totalCertFraction;
  constexpr static int nWheels_ = 5;
  MonitorElement* certWheelFractions[nWheels_];
  constexpr static int nDisks_ = 10;
  MonitorElement* certDiskFractions[nDisks_];
  std::pair<int, int> FEDRange_;
  int numberOfDisks_;
  int NumberOfFeds_;
  bool init_, offlineDQM_;
  double defaultValue_;
};

#endif

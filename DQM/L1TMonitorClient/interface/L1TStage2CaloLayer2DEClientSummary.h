#ifndef DQM_L1TMONITORCLIENT_L1TStage2CaloLayer2DECLIENTSUMMARY_H
#define DQM_L1TMONITORCLIENT_L1TStage2CaloLayer2DECLIENTSUMMARY_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class L1TStage2CaloLayer2DEClientSummary : public DQMEDHarvester {
public:
  L1TStage2CaloLayer2DEClientSummary(const edm::ParameterSet &);

  ~L1TStage2CaloLayer2DEClientSummary() override;

protected:
  void dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                             DQMStore::IGetter &igetter,
                             const edm::LuminosityBlock &lumiSeg,
                             const edm::EventSetup &c) override;

private:
  void book(DQMStore::IBooker &ibooker);
  void processHistograms(DQMStore::IGetter &igetter);

  std::string monitor_dir_;

  MonitorElement *hlSummary;
  MonitorElement *jetSummary;
  MonitorElement *egSummary;
  MonitorElement *tauSummary;
  MonitorElement *sumSummary;
};

#endif

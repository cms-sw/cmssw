#ifndef DQM_RPCMonitorClient_DQMDaqInfo_H
# define DQM_RPCMonitorClient_DQMDaqInfo_H

// system include files
#include <iostream>
#include <fstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"


class RPCDaqInfo : public DQMEDHarvester{
 
public:
  explicit RPCDaqInfo(const edm::ParameterSet&);
  ~RPCDaqInfo();

protected:
  void beginJob() override;
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const&) override; //performed in the endLumi
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

private:
  void  myBooker(DQMStore::IBooker &);

  bool init_;

  MonitorElement*  DaqFraction_;
  MonitorElement * DaqMap_;
  MonitorElement* daqWheelFractions[5];
  MonitorElement* daqDiskFractions[10];

  std::pair<int,int> FEDRange_;

  int  numberOfDisks_,NumberOfFeds_;
 
};

#endif

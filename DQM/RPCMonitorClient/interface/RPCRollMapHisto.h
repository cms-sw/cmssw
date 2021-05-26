#ifndef RPCRollMapHisto_H
#define RPCRollMapHisto_H

#include "DQMServices/Core/interface/DQMStore.h"
#include <string>

struct RPCRollMapHisto {
  typedef dqm::reco::MonitorElement MonitorElement;
  //typedef dqm::impl::MonitorElement MonitorElement;
  //typedef DQMEDHarvester::MonitorElement MonitorElement;
  typedef dqm::reco::DQMStore::IBooker IBooker;

  static MonitorElement* bookBarrel(
      IBooker& booker, const int wheel, const std::string& prefix, const std::string& title, const bool useRollInfo);
  static MonitorElement* bookEndcap(
      IBooker& booker, const int disk, const std::string& prefix, const std::string& title, const bool useRollInfo);

  static void setBarrelRollAxis(MonitorElement* me, const int wheel, const int axis, const bool useRollInfo);
  static void setEndcapRollAxis(MonitorElement* me, const int disk, const int axis, const bool useRollInfo);
};

#endif

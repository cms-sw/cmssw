#ifndef RPCHistoHelper_H
#define RPCHistoHelper_H

#include "DQMServices/Core/interface/DQMStore.h"
#include <string>

struct RPCSummaryMapHisto {
  typedef dqm::implementation::MonitorElement MonitorElement;
  typedef dqm::implementation::IBooker IBooker;

  static MonitorElement* book(IBooker& booker, const std::string& name, const std::string& title);
  static void setBinBarrel(MonitorElement* me, const int wheel, const int sector, const double value);
  static void setBinEndcap(MonitorElement* me, const int disk, const int sector, const double value);
  static void setBinsBarrel(MonitorElement* me, const double value);
  static void setBinsEndcap(MonitorElement* me, const double value);
};

#endif

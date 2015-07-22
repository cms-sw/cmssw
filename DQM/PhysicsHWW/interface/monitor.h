#ifndef WW_monitor_h
#define WW_monitor_h
#include <vector>
#include <string>
#include "wwtypes.h"

#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

using namespace HWWFunctions;


class EventMonitor
{
 public:
  EventMonitor(DQMStore::IBooker& iBooker);
  void count(HypothesisType type, const char* name, double weight=1.0);

  std::map<std::string, int> binMap_;
  MonitorElement *cutflowHist_[4];
};
#endif

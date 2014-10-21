#ifndef WW_monitor_h
#define WW_monitor_h
#include <vector>
#include <string>
#include "wwtypes.h"

using namespace HWWFunctions;


class EventMonitor
{
public:

  EventMonitor();

  struct Entry{
    unsigned int nevt[5];
    std::string name;
    Entry();
  };

  struct hypo_monitor{
    std::vector<EventMonitor::Entry> counters;
    void count(HypothesisType type, const char* name, double weight=1.0);
    hypo_monitor(){}
  };

  hypo_monitor monitor;

};
#endif

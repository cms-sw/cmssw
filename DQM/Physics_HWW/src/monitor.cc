#include "DQM/Physics_HWW/interface/monitor.h"


EventMonitor::Entry::Entry()
{
  for (unsigned int i=0; i<5; ++i){
    nevt[i] = 0;
  }
}

void EventMonitor::hypo_monitor::count(HypothesisType type, const char* name, double weight)
{
  std::vector<EventMonitor::Entry>::iterator itr = counters.begin();
  while (itr != counters.end() && itr->name != name) itr++;
  EventMonitor::Entry* entry(0);
  if ( itr == counters.end() ){
    counters.push_back(Entry());
    entry = &counters.back();
    entry->name = name;
  } else {
    entry = &*itr;
  }
  entry->nevt[type]++;
  entry->nevt[ALL]++;
}

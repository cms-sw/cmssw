#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

HcalBaseMonitor::HcalBaseMonitor() {
  fVerbosity = 0;
}

HcalBaseMonitor::~HcalBaseMonitor() {
}

void HcalBaseMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;
  return;
}

void HcalBaseMonitor::done(){
  return;
}

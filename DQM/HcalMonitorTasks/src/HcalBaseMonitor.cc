#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

HcalBaseMonitor::HcalBaseMonitor() {
  fVerbosity = 0;
  hotCells_.clear();
}

HcalBaseMonitor::~HcalBaseMonitor() {
}

void HcalBaseMonitor::setup(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){
  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;

  hotCells_ =  ps.getUntrackedParameter<vector<string> >( "HotCells" );

  return;
}

void HcalBaseMonitor::done(){
  return;
}

bool HcalBaseMonitor::vetoCell(HcalDetId id){
  if(hotCells_.size()==0) return false;
  for(unsigned int i = 0; i< hotCells_.size(); i++){
    unsigned int badc = atoi(hotCells_[i].c_str());
    if(id.rawId() == badc) return true;
  }
  return false;
}

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

HcalBaseMonitor::HcalBaseMonitor() {
  fVerbosity = 0;
  hotCells_.clear();
  rootFolder_ = "Hcal";
  baseFolder_ = "BaseMonitor";
}

HcalBaseMonitor::~HcalBaseMonitor() {}

void HcalBaseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;

  hotCells_ =  ps.getUntrackedParameter<vector<string> >( "HotCells" );
  
  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Hcal") ;
  rootFolder_ = subsystemname + "/";
  
  fVerbosity = ps.getUntrackedParameter<bool>("debug",0); // shouldn't fVerbosity be an int32?
  makeDiagnostics=ps.getUntrackedParameter<bool>("makeDiagnosticPlots",false);

  return;
}

void HcalBaseMonitor::done(){}

void HcalBaseMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();    
    
    m_dbe->setCurrentFolder(baseFolder_+"/HB");
    m_dbe->removeContents();
    
    m_dbe->setCurrentFolder(baseFolder_+"/HE");
    m_dbe->removeContents();

    m_dbe->setCurrentFolder(baseFolder_+"/HO");
    m_dbe->removeContents();

    m_dbe->setCurrentFolder(baseFolder_+"/HF");
    m_dbe->removeContents();
  }
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

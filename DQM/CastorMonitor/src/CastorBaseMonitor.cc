#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

//***************************************************//
//********** CastorBaseMonitor: *********************//
//********** base class for all monitoring tasks ****//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 29.08.2008          ************// 
//***************************************************//

CastorBaseMonitor::CastorBaseMonitor() {
  fVerbosity = 0;
  hotCells_.clear();
  rootFolder_ = "Castor";
  baseFolder_ = "BaseMonitor";
}

CastorBaseMonitor::~CastorBaseMonitor() {}

void CastorBaseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;

  hotCells_ =  ps.getUntrackedParameter<vector<string> >( "HotCells" );
  
  // Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Castor") ;
  rootFolder_ = subsystemname + "/";
  
  fVerbosity = ps.getUntrackedParameter<bool>("debug",0); 
  makeDiagnostics=ps.getUntrackedParameter<bool>("makeDiagnosticPlots",false);
  showTiming = ps.getUntrackedParameter<bool>("showTiming",false);

  return;
}

void CastorBaseMonitor::done(){}

void CastorBaseMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();    
  }
  return;
}

bool CastorBaseMonitor::vetoCell(HcalCastorDetId id){
  if(hotCells_.size()==0) return false;

  for(unsigned int i = 0; i< hotCells_.size(); i++){
    unsigned int badc = atoi(hotCells_[i].c_str());
    if(id.rawId() == badc) return true;
  }
  return false;
}

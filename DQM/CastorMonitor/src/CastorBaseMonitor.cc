#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

//***************************************************//
//********** CastorBaseMonitor: *********************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 20.08.2008 (first version) *****// 
//***************************************************//
///// Base class for all monitoring tasks

//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorBaseMonitor::CastorBaseMonitor() {
  ////---- parameter to steer debugging messages
  fVerbosity = 0;
  // hotCells_.clear();
  ////---- Define folders
  rootFolder_ = "Castor";
  baseFolder_ = "BaseMonitor";
}

//==================================================================//
//======================= Destructor ===============================//
//==================================================================//
CastorBaseMonitor::~CastorBaseMonitor() {}

void CastorBaseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){

  if(fVerbosity>0) cout << "CastorBaseMonitor::setup (start)" << endl;

  m_dbe = NULL;
  if(dbe != NULL) m_dbe = dbe;

  //hotCells_ =  ps.getUntrackedParameter<vector<string> >( "HotCells" );
  
  ////---- Base folder for the contents of this job
  string subsystemname = ps.getUntrackedParameter<string>("subSystemFolder", "Castor") ;
  rootFolder_ = subsystemname + "/";
  
  fVerbosity = ps.getUntrackedParameter<int>("debug",0); 
  makeDiagnostics=ps.getUntrackedParameter<bool>("makeDiagnosticPlots",false);
  showTiming = ps.getUntrackedParameter<bool>("showTiming",false);

 if(fVerbosity>0) cout << "CastorBaseMonitor::setup (end)" << endl;

  return;
}

//==================================================================//
//============================ done  ===============================//
//==================================================================//
void CastorBaseMonitor::done(){}


//==================================================================//
//=========================== clearME ==============================//
//==================================================================//
void CastorBaseMonitor::clearME(){

  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();    
  }
  return;
}


//==================================================================//
//=========================== vetoCell =============================//
//==================================================================//
bool CastorBaseMonitor::vetoCell(HcalCastorDetId id){
  /*
  if(hotCells_.size()==0) return false;

  for(unsigned int i = 0; i< hotCells_.size(); i++){
    unsigned int badc = atoi(hotCells_[i].c_str());
    if(id.rawId() == badc) return true;
  }
  */
  return false;
}

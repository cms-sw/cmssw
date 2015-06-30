#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"

//***************************************************//
//********** CastorBaseMonitor: *********************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 20.08.2008 (first version) *****// 
///// Base class for all monitoring tasks
//***************************************************//
//---- critical revision 26.06.2014 (Vladimir Popov)
//==================================================================//
//======================= Constructor ==============================//
CastorBaseMonitor::CastorBaseMonitor() {
  fVerbosity = 0;
  rootFolder_ = "Castor";
  baseFolder_ = "BaseMonitor";
  showTiming = false;
}

//======================= Destructor ===============================//
CastorBaseMonitor::~CastorBaseMonitor() {}

//======================= Setup ====================================//
void CastorBaseMonitor::setup(const edm::ParameterSet& ps)
{
  fVerbosity = ps.getUntrackedParameter<int>("debug",0); 
  showTiming = ps.getUntrackedParameter<bool>("showTiming",false);

  if(fVerbosity>0) std::cout << "CastorBaseMonitor::setup (start)" << std::endl;

  
//  pset_ = ps;
  std::string subsystemname = ps.getUntrackedParameter<std::string>("subSystemFolder", "Castor") ;
  rootFolder_ = subsystemname + "/";

  if(fVerbosity>0) std::cout << "CastorBaseMonitor::setup (end)" << std::endl;
  return;
}

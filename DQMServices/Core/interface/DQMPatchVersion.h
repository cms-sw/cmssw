#ifndef _DQM_PATCH_VERSION_H
#define _DQM_PATCH_VERSION_H

#include <string>

namespace dqm {
  std::string DQMPatchVersion = "02" ;
}
#endif


/* 171:01 
 -- List of tags in  DQMPATCH:01 (and submitted for CMSSW_1_7_2)
cvs co -r V00-05-11 DQMServices/Core
cvs co -r V00-05-11 DQMSerivces/CoreROOT
cvs co -r V00-05-11 DQMSerivces/Components
cvs co -r V00-05-11 DQMSerivces/Examples
cvs co -r V00-00-10 DQM/Integration
cvs co -r V00-07-04 DQM/HcalMonitorClient
cvs co -r V00-07-04 DQM/HcalMonitorTasks
cvs co -r V00-07-05 DQM/HcalMonitorModule 
cvs co -r V00-06-00 DQM/RPCMonitorDigi
# -- List of tags in  DQMPATCH:01 (and NOT submitted for CMSSW_1_7_2)
cvs co -r V00-16-05 DQM/DTMonitorModule
cvs co -r V00-03-04 DQM/DTMonitorClient
cvs co -r V02-00-03 DQM/CSCMonitorModule 
cvs co -r V01-08-10 EventFilter/CSCRawToDigi 
*/

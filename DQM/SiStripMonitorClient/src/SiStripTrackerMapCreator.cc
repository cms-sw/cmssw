#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripTrackerMapCreator::SiStripTrackerMapCreator() {
  trackerMap_ = 0;
}
//
// -- Destructor
//
SiStripTrackerMapCreator::~SiStripTrackerMapCreator() {
  if (trackerMap_) delete trackerMap_;
}
//
// -- Create Geometric and Fed Tracker Map
//
void SiStripTrackerMapCreator::create(const edm::ParameterSet & tkmapPset, 
           const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store) {

  if (!trackerMap_)     trackerMap_    = new TrackerMap(tkmapPset, fedcabling);

  const vector<uint16_t>& feds = fedcabling->feds(); 
  uint32_t detId_save = 0;
  map<MonitorElement*,int> local_mes;
  SiStripFolderOrganizer folder_organizer;
  for(vector<unsigned short>::const_iterator ifed = feds.begin(); 
                      ifed < feds.end(); ifed++){
    const std::vector<FedChannelConnection> fedChannels = fedcabling->connections( *ifed );
    for(std::vector<FedChannelConnection>::const_iterator iconn = fedChannels.begin(); iconn < fedChannels.end(); iconn++){
      
      uint32_t detId = iconn->detId();
      if (detId == 0 || detId == 0xFFFFFFFF)  continue;
      if (detId_save != detId) {
        detId_save = detId;
        local_mes.clear();
        string dir_path;
        folder_organizer.getFolderName(detId, dir_path);
        vector<MonitorElement*> all_mes = dqm_store->getContents(dir_path);
	for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	     it!= all_mes.end(); it++) {
          MonitorElement * me = (*it);
          if (!me) continue;
          if (me->getQReports().size() == 0) continue;
          int istat =  SiStripUtility::getMEStatus((*it));
	  local_mes.insert(pair<MonitorElement*, int>((*it), istat));
        }
	paintTkMap(detId,local_mes);              
      }
    }
  }

  trackerMap_->printonline();
}
//
// -- Draw Monitor Elements
//
void SiStripTrackerMapCreator::paintTkMap(int det_id, map<MonitorElement*, int>& me_map) {
  int icol;
  string tag;

  ostringstream comment;
  comment << "Mean Value(s) : ";
  int gstatus = 0;

  MonitorElement* me;
  for (map<MonitorElement*,int>::const_iterator it = me_map.begin(); 
              it != me_map.end(); it++) {
    me = it->first;
    if (!me) continue;
    float mean = me->getMean();
    comment <<   mean <<  " : " ;
    // global status 
    if (it->second > gstatus ) gstatus = it->second;
    SiStripUtility::getMEStatusColor(it->second, icol, tag);   
  }
  if (0) {cout << " Detector ID : " << det_id 
	       << " " << comment.str()
	       << " Status : " << gstatus  << endl;
  }
  trackerMap_->setText(det_id, comment.str());
  int rval, gval, bval;
  SiStripUtility::getMEStatusColor(gstatus, rval, gval, bval);
  trackerMap_->fillc(det_id, rval, gval, bval);
}

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
     const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store, std::string& map_type) {

  if (trackerMap_) delete trackerMap_;
  trackerMap_ = new TrackerMap(tkmapPset, fedcabling);

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
        if (map_type.find("QTestAlarm") != string::npos) paintTkMapFromAlarm(detId, dqm_store);
        else paintTkMapFromHistogram(detId, dqm_store, map_type);
      }
    }
  }

  trackerMap_->printonline();
}
//
// -- Paint Tracker Map with QTest Alarms 
//
void SiStripTrackerMapCreator::paintTkMapFromAlarm(int det_id, DQMStore* dqm_store) {

  SiStripFolderOrganizer folder_organizer;
  string dir_path;
  folder_organizer.getFolderName(det_id, dir_path);
  vector<MonitorElement*> all_mes = dqm_store->getContents(dir_path);

  ostringstream comment;
  int gstatus = 0;

  comment << "Mean Value(s) : ";  
  for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
       it!= all_mes.end(); it++) {
    MonitorElement * me = (*it);
    if (!me) continue;
    if (me->getQReports().size() == 0) continue;
    int istat =  SiStripUtility::getMEStatus((*it));
    comment << me->getMean() << "  " ;
    if (istat > gstatus ) gstatus = istat;
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

//
// --  Paint Tracker Map with Histogram Mean
//
void SiStripTrackerMapCreator::paintTkMapFromHistogram(int det_id, DQMStore* dqm_store, std::string& htype) {

  SiStripFolderOrganizer folder_organizer;
  string dir_path;
  folder_organizer.getFolderName(det_id, dir_path);
  vector<MonitorElement*> all_mes = dqm_store->getContents(dir_path);

  ostringstream comment;
  float fval = 0.0;

  for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
       it!= all_mes.end(); it++) {
    MonitorElement * me = (*it);
    if (!me) continue;
    string hname = me->getName();
    if (hname.find(htype) != string::npos) {
      fval = me->getMean();
      comment << " Mean Value " << fval;
    }
  }
  if (0) {cout << " Detector ID : " << det_id 
	       << " " << comment.str() << endl;
  }
  trackerMap_->fill(det_id, fval);
  trackerMap_->fill_current_val(det_id, fval);
}

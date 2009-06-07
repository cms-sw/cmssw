#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "CalibTracker/SiStripCommon/interface/TkDetMap.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripTrackerMapCreator::SiStripTrackerMapCreator() {
  trackerMap_ = 0;
  if(!edm::Service<TkDetMap>().isAvailable()){
    edm::LogError("TkHistoMap") <<
      "\n------------------------------------------"
      "\nUnAvailable Service TkHistoMap: please insert in the configuration file an instance like"
      "\n\tprocess.TkDetMap = cms.Service(\"TkDetMap\")"
      "\n------------------------------------------";
  }
  tkDetMap_=edm::Service<TkDetMap>().operator->();
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
 
  string tmap_title = " Tracker Map from  " + map_type;
  trackerMap_->setTitle(tmap_title);
  if (map_type != "QTestAlarm") setTkMapRange(map_type);

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
void SiStripTrackerMapCreator::paintTkMapFromAlarm(uint32_t det_id, DQMStore* dqm_store) {
  
  SiStripFolderOrganizer folder_organizer;
  string subdet_folder; 
  folder_organizer.getSubDetFolder(det_id, subdet_folder);
  string badmodule_folder = subdet_folder + "/" + "BadModuleList";

  ostringstream comment;
  comment << " DetId " << det_id << " : ";
  uint16_t flag = 0; 
  if (dqm_store->dirExists(badmodule_folder)) {
    vector<MonitorElement*> all_mes = dqm_store->getContents(badmodule_folder);
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      uint32_t id =  atoi(me->getName().c_str());

      if (id == det_id && me) {
	flag    = me->getIntValue();
      }
    }    
  }

  int rval, gval, bval;
  SiStripUtility::getDetectorStatusColor(flag, rval, gval, bval);
  string message;
  SiStripUtility::getBadModuleStatus(flag, message);
  comment << message.c_str();
  trackerMap_->setText(det_id, comment.str());
  trackerMap_->fillc(det_id, rval, gval, bval);
}

//
// --  Paint Tracker Map with Histogram Mean
//
void SiStripTrackerMapCreator::paintTkMapFromHistogram(uint32_t det_id, DQMStore* dqm_store, std::string& htype) {
  stringstream dir_path;
  SiStripFolderOrganizer folder_organiser;
  folder_organiser.getLayerFolderName(dir_path, det_id);

  vector<MonitorElement*> all_mes = dqm_store->getContents(dir_path.str());
      
  string name = "TkHMap_" + htype;

  MonitorElement * me = 0;
  float fval = 0.0;
  for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
       it!= all_mes.end(); it++) {
    me = (*it);
    if (!me) continue;
    string hname = me->getName();
    if (hname.find(name) != string::npos) {
      break;
    }
  }
  if (me) {
    const TkLayerMap::XYbin& xyval = tkDetMap_->getXY(det_id);
    fval = me->getBinContent(xyval.ix, xyval.iy);
  }

  //  trackerMap_->fill(det_id, fval);

  ostringstream comment;
  comment << " DetId " << det_id << " : ";
  uint16_t flag = 0; 
  string subdet_folder; 
  folder_organiser.getSubDetFolder(det_id, subdet_folder);
  ostringstream badmodule_path;
  badmodule_path << subdet_folder <<  "/" <<  "BadModuleList" << "/" << det_id;
  MonitorElement* bad_module_me = dqm_store->get(badmodule_path.str());
  if (bad_module_me && bad_module_me->kind() == MonitorElement::DQM_KIND_INT) {
    flag = bad_module_me->getIntValue(); 
  }
  string message;  
  SiStripUtility::getBadModuleStatus(flag, message);
  comment << message.c_str();
  trackerMap_->setText(det_id, comment.str());

  trackerMap_->fill_current_val(det_id, fval);
}
//
// -- Get Tracker Map Fill Range
//
void SiStripTrackerMapCreator::setTkMapRange(std::string& map_type) {
  float min = 0.0;
  float max = 0.0; 
  if (map_type.find("FractionOfBadChannels") != string::npos)        max =  1.0;
  else if (map_type.find("NumberOfCluster") != string::npos)         max =  0.01;
  else if (map_type.find("NumberOfDigi") != string::npos)            max =  0.6;
  else if (map_type.find("NumberOfOffTrackCluster") != string::npos) max = 6000.0;
  else if (map_type.find("NumberOfOnTrackCluster") != string::npos)  max = 200.0;
  else if (map_type.find("StoNCorrOnTrack") != string::npos)         max = 200.0;
  trackerMap_->setRange(min, max);
}

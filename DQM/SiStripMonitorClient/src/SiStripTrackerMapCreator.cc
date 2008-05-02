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
  tkMapFrequency_ = -1;
  trackerMap_ = 0;
}
//
// -- Destructor
//
SiStripTrackerMapCreator::~SiStripTrackerMapCreator() {
  if (trackerMap_) delete trackerMap_;
}
//
// -- Read ME list
//
bool SiStripTrackerMapCreator::readConfiguration() {
  SiStripConfigParser config_parser;
  string localPath = string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  config_parser.getDocument(edm::FileInPath(localPath).fullPath());
  if (!config_parser.getFrequencyForTrackerMap(tkMapFrequency_)){
    cout << "SiStripActionExecutor::readConfiguration: Failed to read TrackerMap configuration parameters!! ";
    tkMapFrequency_ = -1;
    return false;
  }

  if (!config_parser.getMENamesForTrackerMap(tkMapName_, meNames_)){  
    cout << "SiStripTrackerMapCreator::readConfiguration: Failed to read TrackerMap configuration parameters!! ";
    return false;
  }
  cout << " # of MEs in Tk Map " << meNames_.size() << endl;
  return true;
}
//
// -- Get Tracker Map ME names
//
int SiStripTrackerMapCreator::getMENames(vector<string>& me_names) {
  if (meNames_.size() == 0) return 0;
  for (vector<string>::const_iterator im = meNames_.begin();
       im != meNames_.end(); im++) {
    me_names.push_back(*im);
  }
  return me_names.size();
}
//
// -- Create Geometric and Fed Tracker Map
//
void SiStripTrackerMapCreator::create(const edm::ParameterSet & tkmapPset, 
           const edm::ESHandle<SiStripFedCabling>& fedcabling, DQMStore* dqm_store) {

  if (meNames_.size() == 0) return;
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
	  if (!(*it)) continue;
	  string me_name = (*it)->getName();        
	  int istat = 0;
	  for (vector<string>::const_iterator im = meNames_.begin();
	       im != meNames_.end(); im++) {
	    if (me_name.find(*im) == string::npos) continue;
	    istat =  SiStripUtility::getMEStatus((*it)); 
	    local_mes.insert(pair<MonitorElement*, int>((*it), istat));
	  }
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

#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMap.h"
#include "CommonTools/TrackerMap/interface/FedTrackerMap.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include <iostream>
#include "TText.h"
using namespace std;
//
// -- Constructor
// 
SiStripTrackerMapCreator::SiStripTrackerMapCreator() {
  tkMapFrequency_ = -1;
  trackerMap_ = 0;
  fedTrackerMap_ = 0;
}
//
// -- Destructor
//
SiStripTrackerMapCreator::~SiStripTrackerMapCreator() {
  if (trackerMap_) delete trackerMap_;
  if (fedTrackerMap_) delete fedTrackerMap_;
}
//
// -- Read ME list
//
bool SiStripTrackerMapCreator::readConfiguration() {
  SiStripConfigParser config_parser;
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_monitorelement_config.xml");
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
// -- Browse through monitorable and get values need for TrackerMap
//
void SiStripTrackerMapCreator::create(DaqMonitorBEInterface* bei) {
  if (meNames_.size() == 0) return;
  if (!trackerMap_) trackerMap_ = new SiStripTrackerMap(tkMapName_);

  vector<string> tempVec, contentVec;
  bei->getContents(tempVec);
  for (vector<string>::iterator it = tempVec.begin();
       it != tempVec.end(); it++) {
    if ((*it).find("module_") != string::npos) contentVec.push_back(*it);
  }
  tempVec.clear();
  int ibin = 0;
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    ibin++;
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    // get module id
    string det_id = ((*it).substr((*it).find("module_")+7, 9)).c_str();
    
    map<MonitorElement*,int> local_mes;
    int gstat = 0;
    //  browse through monitorable; check  if required MEs exist    
    for (vector<string>::const_iterator ic = contents.begin();
	      ic != contents.end(); ic++) {
      int istat = 0;
      for (vector<string>::const_iterator im = meNames_.begin();
	   im != meNames_.end(); im++) {
        string me_name = (*im);
	if ((*ic).find(me_name) == string::npos) continue;
        MonitorElement * me = bei->get((*ic));
        if (!me) continue;
        istat =  SiStripUtility::getStatus(me); 
        local_mes.insert(pair<MonitorElement*, int>(me, istat));
	if (istat > gstat) gstat = istat;
      }
    }
    paintTkMap(atoi(det_id.c_str()), local_mes);
  }
  trackerMap_->printonline();  
}
//
// -- Create Fed Tracker Map
//
void SiStripTrackerMapCreator::createFedTkMap(const edm::ESHandle<SiStripFedCabling> fedcabling, DaqMonitorBEInterface* bei) {

  if (meNames_.size() == 0) return;
  if (! fedTrackerMap_ )   fedTrackerMap_ = new FedTrackerMap(fedcabling);

  const vector<uint16_t>& feds = fedcabling->feds(); 
  for(vector<unsigned short>::const_iterator ifed = feds.begin(); 
                      ifed < feds.end(); ifed++){
    const std::vector<FedChannelConnection> fedChannels = fedcabling->connections( *ifed );
    for(std::vector<FedChannelConnection>::const_iterator iconn = fedChannels.begin(); iconn < fedChannels.end(); iconn++){
      
      uint32_t detId = iconn->detId();
      if (detId == 0) continue;
      vector<MonitorElement*> all_mes = bei->get(detId);
      map<MonitorElement*,int> local_mes;
      for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	   it!= all_mes.end(); it++) {
	if (!(*it)) continue;
	string me_name = (*it)->getName();        
	int istat = 0;
	for (vector<string>::const_iterator im = meNames_.begin();
	     im != meNames_.end(); im++) {
	  if (me_name.find(*im) == string::npos) continue;
	  istat =  SiStripUtility::getStatus((*it)); 
	  local_mes.insert(pair<MonitorElement*, int>((*it), istat));
        }
      }
      paintFedTkMap(iconn->fedId(), iconn->fedCh(),local_mes);      
    }
  }
  fedTrackerMap_->print();  
}
//
// -- Paint FED Tracker Map
//
void SiStripTrackerMapCreator::paintFedTkMap(int fed_id, int fed_ch, std::map<MonitorElement*, int>& me_map) {
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
    SiStripUtility::getStatusColor(it->second, icol, tag);   
  }
  cout << " FedId  : " << fed_id << " Fed Ch. " << fed_ch
       << " " << comment.str()
       << " Status : " << gstatus  << endl;
  
  fedTrackerMap_->setText(fed_id, fed_ch, comment.str());
  int rval, gval, bval;
  SiStripUtility::getStatusColor(gstatus, rval, gval, bval);
  fedTrackerMap_->fillc(fed_id, fed_ch, rval, gval, bval);
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
void SiStripTrackerMapCreator::create(const edm::ESHandle<SiStripFedCabling> fedcabling, DaqMonitorBEInterface* bei) {

  if (meNames_.size() == 0) return;
  if (!trackerMap_)     trackerMap_    = new SiStripTrackerMap(tkMapName_);
  if (!fedTrackerMap_ ) fedTrackerMap_ = new FedTrackerMap(fedcabling);

  const vector<uint16_t>& feds = fedcabling->feds(); 
  uint32_t detId_save = 0;
  map<MonitorElement*,int> local_mes;
  for(vector<unsigned short>::const_iterator ifed = feds.begin(); 
                      ifed < feds.end(); ifed++){
    const std::vector<FedChannelConnection> fedChannels = fedcabling->connections( *ifed );
    for(std::vector<FedChannelConnection>::const_iterator iconn = fedChannels.begin(); iconn < fedChannels.end(); iconn++){
      
      uint32_t detId = iconn->detId();
      if (detId == 0) continue;
      if (detId_save != detId) {
        detId_save = detId;
        local_mes.clear();
        vector<MonitorElement*> all_mes = bei->get(detId);
	for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	     it!= all_mes.end(); it++) {
	  if (!(*it)) continue;
	  string me_name = (*it)->getName();        
	  int istat = 0;
	  for (vector<string>::const_iterator im = meNames_.begin();
	       im != meNames_.end(); im++) {
	    if (me_name.find(*im) == string::npos) continue;
	    istat =  SiStripUtility::getStatus((*it)); 
	    local_mes.insert(pair<MonitorElement*, int>((*it), istat));
	  }
        }
	paintTkMap(detId,local_mes);              
      }
      paintFedTkMap(iconn->fedId(), iconn->fedCh(),local_mes);      
    }
  }
  trackerMap_->printonline();
  fedTrackerMap_->print();  
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
    SiStripUtility::getStatusColor(it->second, icol, tag);   
  }
  cout << " Detector ID : " << det_id 
       << " " << comment.str()
       << " Status : " << gstatus  << endl;
  
  trackerMap_->setText(det_id, comment.str());
  int rval, gval, bval;
  SiStripUtility::getStatusColor(gstatus, rval, gval, bval);
  trackerMap_->fillc(det_id, rval, gval, bval);
}

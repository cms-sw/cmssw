#include "DQM/SiStripMonitorClient/interface/TrackerMapCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/QTestStatus.h"
#include <iostream>
#include "TText.h"
using namespace std;
//
// -- Constructor
// 
TrackerMapCreator::TrackerMapCreator() {
  trackerMap = new TrackerMap("SiStripMap");
}
//
// -- Destructor
//
TrackerMapCreator::~TrackerMapCreator() {
  if (trackerMap) delete trackerMap;
}
//
// -- Browse through monitorable and get values need for TrackerMap
//
void TrackerMapCreator::create(MonitorUserInterface* mui, vector<string>& me_names) {

  vector<string> tempVec, contentVec;
  mui->getContents(tempVec);
  for (vector<string>::iterator it = tempVec.begin();
       it != tempVec.end(); it++) {
    if ((*it).find("module_") != string::npos) contentVec.push_back(*it);
  }
  int ndet = contentVec.size();
  tempVec.clear();
  int ibin = 0;
  string gname = "GobalFlag";
  MonitorElement* tkmap_gme = getTkMapMe(mui,gname,ndet); 
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
      for (vector<string>::const_iterator im = me_names.begin();
	   im != me_names.end(); im++) {
        string me_name = (*im);
	if ((*ic).find(me_name) == string::npos) continue;
        MonitorElement * me = mui->get((*ic));
        if (!me) continue;
        istat =  SiStripUtility::getStatus(me); 
        local_mes.insert(pair<MonitorElement*, int>(me, istat));
	if (istat > gstat) gstat = istat;
        MonitorElement* tkmap_me = getTkMapMe(mui,me_name,ndet);
	if (tkmap_me){
          tkmap_me->Fill(ibin, istat);
	  tkmap_me->setBinLabel(ibin, det_id.c_str());
        }
      }
    }
    if (tkmap_gme) {
       tkmap_gme->Fill(ibin, gstat);
       tkmap_gme->setBinLabel(ibin, det_id.c_str());
    }    
    paintTkMap(atoi(det_id.c_str()), local_mes);
  }
  trackerMap->print(true);  
}
//
// -- Draw Monitor Elements
//
void TrackerMapCreator::paintTkMap(int det_id, map<MonitorElement*, int>& me_map) {
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
  
  trackerMap->setText(det_id, comment.str());
  int rval, gval, bval;
  SiStripUtility::getStatusColor(gstatus, rval, gval, bval);
  trackerMap->fillc(det_id, rval, gval, bval);
}
//
// -- get Tracker Map ME 
//
MonitorElement* TrackerMapCreator::getTkMapMe(MonitorUserInterface* mui, 
                    string& me_name, int ndet) {
  string new_name = "TrackerMap_for_" + me_name;
  string path = "Collector/" + new_name;
  MonitorElement*  tkmap_me =0;
  tkmap_me = mui->get(path);
  if (!tkmap_me) {
    string save_dir = mui->pwd();   
    DaqMonitorBEInterface * bei = mui->getBEInterface();
    bei->setCurrentFolder("Collector");
    tkmap_me = bei->book1D(new_name, new_name, ndet, 0.5, ndet+0.5);
    bei->setCurrentFolder(save_dir);
  }
  return tkmap_me;
}

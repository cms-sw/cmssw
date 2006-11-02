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
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    ibin++;
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    // get module id
    string det_id = ((*it).substr((*it).find("module_")+7)).c_str();
    
    map<int, MonitorElement*> local_mes;
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
        istat =  getStatus(me); 
        local_mes.insert(pair<int, MonitorElement*>(istat, me));
	
        MonitorElement* tkmap_me = getTkMapMe(mui,me_name,ndet);
	if (tkmap_me)  tkmap_me->Fill(ibin, istat);
	tkmap_me->setBinLabel(ibin, det_id.c_str());
      }
    }
    drawMEs(atoi(det_id.c_str()), local_mes);
  }
  trackerMap->print(true);  
}
//
// -- Draw Monitor Elements
//
void TrackerMapCreator::drawMEs(int det_id, map<int, MonitorElement*>& me_map) {

  TCanvas canvas("display");
  canvas.Clear();
  if (me_map.size() == 2) canvas.Divide(1,2);
  if (me_map.size() == 3) canvas.Divide(1,3);
  if (me_map.size() == 4) canvas.Divide(2,2);

  int icol;
  string tag;

  ostringstream comment;
  comment << "Mean Value(s) : ";
  int gstatus = 0;

  MonitorElement* me;
  int i = 0;
  for (map<int, MonitorElement*>::const_iterator it = me_map.begin(); 
              it != me_map.end(); it++) {
    i++;
    me = it->second;
    if (!me) continue;
    float mean = me->getMean();
    comment <<   mean <<  " : " ;
    // global status 
    if (it->first > gstatus ) gstatus = it->first;

    getStatusColor(it->first, icol, tag);
   
    // Access the Root object and plot
    MonitorElementT<TNamed>* ob = 
        dynamic_cast<MonitorElementT<TNamed>*>(me);
    if (ob) {
      canvas.cd(i);
      TText tt;
      tt.SetTextSize(0.15);
      tt.SetTextColor(icol);
      ob->operator->()->Draw();
      tt.DrawTextNDC(0.6, 0.5, tag.c_str());
      canvas.Update();
    }
  }

  cout << " Detector ID : " << det_id 
       << " " << comment.str()
       << " Status : " << gstatus  << endl;
  
  trackerMap->setText(det_id, comment.str());
  int rval, gval, bval;
  getStatusColor(gstatus, rval, gval, bval);
  trackerMap->fillc(det_id, rval, gval, bval);

    
  ostringstream name_str;
  name_str << det_id << ".jpg";
  canvas.SaveAs(name_str.str().c_str());    
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
//
// -- Get Color code from Status
//
void TrackerMapCreator::getStatusColor(int status, int& rval, int&gval, int& bval) {
  if (status == dqm::qstatus::STATUS_OK) { 
    rval = 0;   gval = 255;   bval = 0; 
  } else if (status == dqm::qstatus::WARNING) { 
    rval = 255; gval = 255; bval = 0;
  } else if (status == dqm::qstatus::ERROR) { 
    rval = 255; gval = 0;  bval = 0;
  } else if (status == dqm::qstatus::OTHER) { 
    rval = 255; gval = 150;  bval = 0;
  } else {
    rval = 0; gval = 0;  bval = 255;
  }        
}
//
// -- Get Color code from Status
//
void TrackerMapCreator::getStatusColor(int status, int& icol, string& tag) {
  if (status == dqm::qstatus::STATUS_OK) { 
    tag = "Ok";
    icol = 3;
  } else if (status == dqm::qstatus::WARNING) { 
    tag = "Warning";
    icol = 5;     
  } else if (status == dqm::qstatus::ERROR) { 
    tag = "Error";
    icol = 2;
  } else if (status == dqm::qstatus::OTHER) { 
    tag = "Other";
    icol = 1;
  } else {
    tag = " ";
    icol = 1;
  }     
}
//
// -- Get Status of Monitor Element
//
int TrackerMapCreator::getStatus(MonitorElement* me) {
  int status = 0; 
  if (me->getQReports().size() == 0) {
    status = 0;
  } else if (me->hasError()) {
    status = dqm::qstatus::ERROR;
  } else if (me->hasWarning()) {
    status = dqm::qstatus::WARNING;
  } else if (me->hasOtherReport()) {
    status = dqm::qstatus::OTHER;
  } else {  
    status = dqm::qstatus::STATUS_OK;
  }
  return status;
}

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMap.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TText.h"
#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripActionExecutor::SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Creating SiStripActionExecutor " << "\n" ;
  configParser_ = 0;
  configWriter_ = 0;
}
//
// --  Destructor
// 
SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripActionExecutor " << "\n" ;
  if (configParser_) delete configParser_;
}
//
// -- Read Configurationn File
//
void SiStripActionExecutor::readConfiguration() {
  if (configParser_ == 0) {
    configParser_ = new SiStripConfigParser();
    configParser_->getDocument("sistrip_monitorelement_config.xml");
  }
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration(int& tkmap_freq, int& sum_freq) {
  if (configParser_ == 0) {
    configParser_ = new SiStripConfigParser();
    configParser_->getDocument("sistrip_monitorelement_config.xml");
  }
  if (!configParser_->getFrequencyForTrackerMap(tkmap_freq)){
    cout << "SiStripActionExecutor::readConfiguration: Failed to read TrackerMap configuration parameters!! ";
    return false;
  }
  if (!configParser_->getFrequencyForSummary(sum_freq)){
    cout << "SiStripActionExecutor::readConfiguration: Failed to read Summary configuration parameters!! ";
    return false;
  }
  return true;
}
//
// -- Create Tracker Map
//
void SiStripActionExecutor::createTkMap(MonitorUserInterface* mui) {
  string tkmap_name;
  vector<string> me_names;
  if (!configParser_->getMENamesForTrackerMap(tkmap_name, me_names)){
    cout << "SiStripActionExecutor::createTkMap: Failed to read TrackerMap configuration parameters!! ";
    return;
  }
  cout << " # of MEs in Tk Map " << me_names.size() << endl;
  TrackerMap trackerMap(tkmap_name);
  // Get the values for the Tracker Map
  mui->cd();
  SiStripActionExecutor::DetMapType valueMap;
  //  DaqMonitorBEInterface * bei = mui->getBEInterface();
  //  bei->lock();
  getValuesForTkMap(mui, me_names, valueMap);  
  //  bei->unlock();
  int rval = 0;
  int gval = 0;
  int bval = 0;
  for (SiStripActionExecutor::DetMapType::const_iterator it = valueMap.begin();
       it != valueMap.end(); it++) {
    if (it->second.size() < 1) continue;
    int istat = 0;
    ostringstream comment;
    comment << "Mean Value(s) : ";
    for (vector<pair <int,float> >::const_iterator iv = it->second.begin();
	 iv != it->second.end();  iv++) {
      if (iv->first > istat ) istat = iv->first;
      comment <<   iv->second <<  " : " ;
     // Fill Tracker Map with Mean Value of the first element
      if (iv == it->second.begin()) trackerMap.fill_current_val(it->first, iv->first);
    }

   // Fill Tracker Map with color from the status
    if (istat == dqm::qstatus::STATUS_OK) { 
      rval = 0;   gval = 255;   bval = 0; 
    } else if (istat == dqm::qstatus::WARNING) { 
      rval = 255; gval = 255; bval = 0;
    } else if (istat == dqm::qstatus::ERROR) { 
      rval = 255; gval = 0;  bval = 0;
    }
    cout << " Detector ID : " << it->first 
	 << comment.str()
         << " Status : " << istat << endl;
    trackerMap.fillc(it->first, rval, gval, bval);
    // Fill Tracker Map with Mean Value as Comment
    trackerMap.setText(it->first, comment.str());
  }
  trackerMap.print(true);
  return;
}
//
// -- Browse through monitorable and get values need for TrackerMap
//
void SiStripActionExecutor::getValuesForTkMap(MonitorUserInterface* mui,
 vector<string> me_names, SiStripActionExecutor::DetMapType & values) {
  string currDir = mui->pwd();
  vector<string> contentVec;
  mui->getContents(contentVec);

  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    if ((*it).find("module_") == string::npos) continue;
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    // get module id
    int id = atoi(((*it).substr((*it).find("module_")+7)).c_str());
    vector<MonitorElement*> me_vec;
    vector<pair <int, float> > vtemp;
    
    //  browse through monitorable; check  if required MEs exist    
    for (vector<string>::const_iterator ic = contents.begin();
	      ic != contents.end(); ic++) {
      for (vector<string>::const_iterator im = me_names.begin();
	   im != me_names.end(); im++) {
	if ((*ic).find((*im)) == string::npos) continue;

        MonitorElement * me = mui->get((*ic));
        if (me) me_vec.push_back(me);
      }
    }
    drawMEs(id, me_vec, vtemp);
    values.insert(pair<int,vector <pair <int,float> > >(id, vtemp));
  }
}
// -- Browse through the Folder Structure
//
void SiStripActionExecutor::createSummary(MonitorUserInterface* mui) {
  string structure_name;
  vector<string> me_names;
  if (!configParser_->getMENamesForSummary(structure_name, me_names)) {
    cout << "SiStripActionExecutor::createSummary: Failed to read Summary configuration parameters!! ";
    return;
  }
  mui->cd();
  fillSummary(mui, structure_name, me_names);
  mui->cd();
  createLayout(mui);
  string fname = "test.xml";
  configWriter_->write(fname);
  if (configWriter_) delete configWriter_;
  configWriter_ = 0;
}
//
// -- Browse through the Folder Structure
//
void SiStripActionExecutor::fillSummary(MonitorUserInterface* mui,
                               string dir_name,vector<string>& me_names) {
  string currDir = mui->pwd();
  if (currDir.find(dir_name) != string::npos)  {
    vector<MonitorElement*> sum_mes;
    for (vector<string>::const_iterator iv = me_names.begin();
	 iv != me_names.end(); iv++) {
      string tag = "Summary_" + (*iv) + "_in_" 
                                + currDir.substr(currDir.find(dir_name));
      MonitorElement* temp = getSummaryME(mui, tag);
      sum_mes.push_back(temp);
    }
    if (sum_mes.size() == 0) {
      cout << " Summary MEs can not be created" << endl;
      return;
    }
    vector<string> subdirs = mui->getSubdirs();
    int ndet = 0;
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("module_") == string::npos) continue;
      mui->cd(*it);
      ndet++;
      vector<string> contents = mui->getMEs();    
      for (vector<MonitorElement*>::const_iterator isum = sum_mes.begin();
	   isum != sum_mes.end(); isum++) {
	for (vector<string>::const_iterator im = contents.begin();
	     im != contents.end(); im++) {
          string sname = ((*isum)->getName());
          string tname = sname.substr(8,(sname.find("_",8)-8));
	  if (((*im)).find(tname) == 0) {
	    string fullpathname = mui->pwd() + "/" + (*im); 
	    MonitorElement *  me = mui->get(fullpathname);
	    if (me) (*isum)->Fill(ndet*1.0, me->getMean());
            break;
          }
	}
      }
      mui->goUp();
    }    
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      mui->cd(*it);
      fillSummary(mui, dir_name, me_names);
      mui->goUp();
    }
  }
}
//
// -- Get Summary ME
//
MonitorElement* SiStripActionExecutor::getSummaryME(MonitorUserInterface* mui,
                                               string me_name) {
  MonitorElement* me = 0;
  // If already booked
  vector<string> contents = mui->getMEs();    
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      string fullpathname = mui->pwd() + "/" + (*it); 
      me = mui->get(fullpathname);
      if (me) {
	MonitorElementT<TNamed>* obh1 = 
	  dynamic_cast<MonitorElementT<TNamed>*> (me);
	if (obh1) {
	  TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	  if (root_obh1) root_obh1->Reset();        
	}
	return me;
      }
    }
  }
  DaqMonitorBEInterface * bei = mui->getBEInterface();
  me = bei->book1D(me_name.c_str(), me_name.c_str(),20,0.5,20.5);
  return me;
}
//
// -- Draw Monitor Elements
//
void SiStripActionExecutor::drawMEs(int idet, 
  vector<MonitorElement*>& mon_elements, vector<pair <int, float> > & values) {
  TCanvas canvas("display");
  canvas.Clear();
  if (mon_elements.size() == 2) canvas.Divide(1,2);
  if (mon_elements.size() == 3) canvas.Divide(1,3);
  if (mon_elements.size() == 4) canvas.Divide(2,2);
  int status;
  int icol;
  string tag;
  
  for (unsigned int i = 0; i < mon_elements.size(); i++) {
    // Mean Value
    float mean_val = mon_elements[i]->getMean();
    // Status after comparison to Referece 
    if (mon_elements[i]->hasError()) {
      status = dqm::qstatus::ERROR;
      tag = "Error";
      icol = 2;
    } else if (mon_elements[i]->hasWarning()) {
      status = dqm::qstatus::WARNING;
      tag = "Warning";
      icol = 5;
    } else  {
      status = dqm::qstatus::STATUS_OK;
      tag = "Ok";
      icol = 3;
    }
    // Access the Root object and plot
    MonitorElementT<TNamed>* ob = 
        dynamic_cast<MonitorElementT<TNamed>*>(mon_elements[i]);
    if (ob) {
      canvas.cd(i+1);
      TText tt;
      tt.SetTextSize(0.15);
      tt.SetTextColor(icol);
      ob->operator->()->Draw();
      tt.DrawTextNDC(0.6, 0.5, tag.c_str());
      canvas.Update();
    }
    values.push_back(pair<int,float>(status, mean_val));
  }
  ostringstream name_str;
  name_str << idet << ".jpg";
  canvas.SaveAs(name_str.str().c_str());    
  
}
//
// -- Setup Quality Tests 
//
void SiStripActionExecutor::setupQTests(MonitorUserInterface * mui) {
  SiStripQualityTester qtester;
  qtester.setupQTests(mui);
}
//
// -- Check Status of Quality Tests
//
void SiStripActionExecutor::checkQTestResults(MonitorUserInterface * mui) {
  string currDir = mui->pwd();
  vector<string> contentVec;
  mui->getContents(contentVec);
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      MonitorElement * me = mui->get((*im));
      if (me) {
	// get all warnings associated with me
	vector<QReport*> warnings = me->getQWarnings();
	for(vector<QReport *>::const_iterator it = warnings.begin();
	    it != warnings.end(); ++it) {
	  edm::LogWarning("SiStripQualityTester::checkTestResults") << 
	    " *** Warning for " << me->getName() << 
	    "," << (*it)->getMessage() << "\n";
	  
	  cout <<  " *** Warning for " << me->getName() << "," 
	       << (*it)->getMessage() << " " << me->getMean() 
	       << " " << me->getRMS() << me->hasWarning() 
	       << endl;
	}
	// get all errors associated with me
	vector<QReport *> errors = me->getQErrors();
	for(vector<QReport *>::const_iterator it = errors.begin();
	    it != errors.end(); ++it) {
	  edm::LogError("SiStripQualityTester::checkTestResults") << 
	    " *** Error for " << me->getName() << 
	    "," << (*it)->getMessage() << "\n";
	  
	  cout  <<   " *** Error for " << me->getName() << ","
		<< (*it)->getMessage() << " " << me->getMean() 
		<< " " << me->getRMS() 
		<< endl;
	}
      }
    }
  }
}
//
//
//
void SiStripActionExecutor::createCollation(MonitorUserInterface * mui){
   string currDir = mui->pwd();
   
  vector<string> contentVec;
  mui->getContents(contentVec);

  for (vector<string>::iterator it = contentVec.begin();
      it != contentVec.end(); it++) {
    if ((*it).find("module_") == string::npos) continue;
    string dir_path;
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), dir_path, contents);
    string coll_dir = "Collector/Collated/"
             +dir_path.substr(dir_path.find("SiStrip"),dir_path.size());
    cout << " Here " << coll_dir << " ==> " << endl;
    for (vector<string>::iterator ic = contents.begin(); ic != contents.end(); ic++) {
      CollateMonitorElement* sum=mui->collate1D((*ic),(*ic),coll_dir);
      string me_path = dir_path + (*ic);
      mui->add(sum, me_path);

    }
  }
}
void SiStripActionExecutor::createLayout(MonitorUserInterface * mui){
  if (configWriter_ == 0) {
    configWriter_ = new SiStripConfigWriter();
    if (!configWriter_->init()) return;
  }
  string currDir = mui->pwd();   
  if (currDir.find("layer") != string::npos) {
    string name = "Default";
   configWriter_->createLayout(name);
   configWriter_->createRow();
    fillLayout(mui);
  } else {
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      createLayout(mui);
      mui->goUp();
    }
  }
  
}
void SiStripActionExecutor::fillLayout(MonitorUserInterface * mui){
  
  static int icount = 0;
  string currDir = mui->pwd();
  if (currDir.find("string_") != string::npos) {
    vector<string> contents = mui->getMEs(); 
    for (vector<string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
      if ((*im).find("Clusters") != string::npos) {
        icount++;
        if (icount != 0 && icount%6 == 0) {
          configWriter_->createRow();
        }
        ostringstream full_path;
	full_path << "test/" << currDir << "/" << *im ;
        string element = "monitorable";
        string element_name = full_path.str();     
        configWriter_->createColumn(element, element_name);
      }
    }
  } else {
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      fillLayout(mui);
      mui->goUp();
    }
  }
}

#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TText.h"
#include <iostream>

//
// -- Constructor
// 
SiStripActionExecutor::SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Creating SiStripActionExecutor " << "\n" ;
}
//
// --  Destructor
// 
SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripActionExecutor " << "\n" ;
}
//
// -- Create Tracker Map
//
void SiStripActionExecutor::createTkMap(MonitorUserInterface* mui,
			std::vector<std::string> me_names) {
  TrackerMap trackerMap("SiStrip Map");
  // Get the values for the Tracker Map
  mui->cd();
  SiStripActionExecutor::DetMapType valueMap;
  getValuesForTkMap(mui, me_names, valueMap);  
  int rval = 0;
  int gval = 0;
  int bval = 0;
  for (SiStripActionExecutor::DetMapType::const_iterator it = valueMap.begin();
       it != valueMap.end(); it++) {
    if (it->second.size() < 1) continue;
    int istat = 0;
    ostringstream comment;
    comment << "Mean Value(s) : ";
    for (std::vector<std::pair <int,float> >::const_iterator iv = it->second.begin();
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
 std::vector<std::string> me_names, SiStripActionExecutor::DetMapType & values) {
  std::string currDir = mui->pwd();
  // browse through monitorable; check  if MEs exist
  if (currDir.find("module_") != std::string::npos)  {
    // Geometric ID
    int id = atoi((currDir.substr(currDir.find("module_")+7)).c_str());
    TCanvas canvas("display");
    canvas.Clear();
    if (me_names.size() == 2) canvas.Divide(1,2);
    if (me_names.size() == 3) canvas.Divide(1,3);
    if (me_names.size() == 4) canvas.Divide(2,2);
    int idiv = 0;

    int status;
    int icol;
    string tag;
    std::vector<std::string> contents = mui->getMEs();    

    std::vector<std::pair <int, float> > vtemp;
    for (std::vector<std::string>::const_iterator im = me_names.begin();
	 im != me_names.end(); im++) {
      idiv++;
      
      for (std::vector<std::string>::const_iterator it = contents.begin();
	      it != contents.end(); it++) {
	if ((*it).find((*im)) != 0) continue;
	std::string fullpathname = currDir + "/" + (*it); 
        MonitorElement * me = mui->get(fullpathname);
        if (me) {
          // Mean Value
	  float mean_val = me->getMean();
          // Status after comparison to Referece 
	  if (me->hasError()) {
            status = dqm::qstatus::ERROR;
            tag = "Error";
            icol = 2;
	  } else if (me->hasWarning()) {
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
	    dynamic_cast<MonitorElementT<TNamed>*>(me);
	  if (ob) {
            canvas.cd(idiv);
            TText tt;
            tt.SetTextSize(0.06);
            tt.SetTextColor(icol);
	    ob->operator->()->Draw();
            tt.DrawTextNDC(0.7, 0.5, tag.c_str());
            canvas.Update();
          }
          vtemp.push_back(std::pair<int,float>(status, mean_val));
        }
      }
    }
    values.insert(pair<int,std::vector <std::pair <int,float> > >(id, vtemp));
    ostringstream name_str;
    name_str << id << ".jpg";
    canvas.SaveAs(name_str.str().c_str());    
  } else {
    std::vector<std::string> subdirs = mui->getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      getValuesForTkMap(mui, me_names, values);
      mui->goUp();
    }
  } 
}
//
// -- Browse through the Folder Structure
//
void SiStripActionExecutor::fillSummary(MonitorUserInterface* mui,
                     std::string dir_name, std::string me_name) {
  std::string currDir = mui->pwd();

  if (currDir.find(dir_name) != std::string::npos)  {
    std::string tag = "Summary" + me_name + "_in_" 
                                + currDir.substr(currDir.find(dir_name));
    MonitorElement* sum_me = getSummaryME(mui, tag);
    if (!sum_me) return;
    std::vector<std::string> subdirs = mui->getSubdirs();
    int ndet = 0;
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("module_") == std::string::npos) continue;
      mui->cd(*it);
      ndet++;
      std::vector<std::string> contents = mui->getMEs();    
      for (std::vector<std::string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
	if ((*im).find(me_name) == 0) {
	  std::string fullpathname = mui->pwd() + "/" + (*im); 
	  MonitorElement *  me = mui->get(fullpathname);
	  if (me) sum_me->Fill(ndet*1.0, me->getMean());
	}
      }
      mui->goUp();
    }    
  } else {  
    std::vector<std::string> subdirs = mui->getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      mui->cd(*it);
      fillSummary(mui, dir_name, me_name);
      mui->goUp();
    }
  }
}
//
// -- Get Summary ME
//
MonitorElement* SiStripActionExecutor::getSummaryME(MonitorUserInterface* mui,
                                               std::string me_name) {
  MonitorElement* me = 0;
  // If already booked
  std::vector<std::string> contents = mui->getMEs();    
  for (std::vector<std::string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(me_name) == 0) {
      std::string fullpathname = mui->pwd() + "/" + (*it); 
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
// Check Status of Quality Tests
void SiStripActionExecutor::checkTestResults(MonitorUserInterface * mui) {
  string currDir = mui->pwd();
  // browse through monitorables and check test results
  std::vector<std::string> subdirs = mui->getSubdirs();
  std::vector<string> contents = mui->getMEs();    
  for (std::vector<string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
    std::string fullpathname = currDir + "/" + (*it); 
    MonitorElement * me = mui->get(fullpathname);
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
  for (std::vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    mui->cd(*it);
    checkTestResults(mui);
    mui->goUp();
  } 
}

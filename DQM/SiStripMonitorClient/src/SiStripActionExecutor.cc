#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
					std::string me_name) {
  TrackerMap trackerMap(me_name);
  // Get the values for the Tracker Map
  mui->cd();
  SiStripActionExecutor::DetMapType valueMap;
  getValuesForTkMap(mui, me_name, valueMap);  
  std::string comment;
  int rval = 0;
  int gval = 0;
  int bval = 0;
  for (SiStripActionExecutor::DetMapType::const_iterator it = valueMap.begin();
       it != valueMap.end(); it++) {
    if (it->second.size() < 2) continue;
    cout << " Detector ID : " << it->first 
	 << " Mean Value : " << it->second[1] 
         << " Status : " << it->second[0]  << endl;
    // Fill Tracker Map with color from the status
    if (it->second[0] == "Ok") { 
      rval = 0;   gval = 255;   bval = 0; 
    } else if (it->second[0] == "Warning") { 
      rval = 255; gval = 255; bval = 0;
    } else if (it->second[0] == "Error") { 
      rval = 255; gval = 0;  bval = 0;
    }
    trackerMap.fillc(it->first, rval, gval, bval);
    // Fill Tracker Map with Mean Value
    trackerMap.fill_current_val(it->first, atof(it->second[1].c_str()));
    // Fill Tracker Map with Mean Value as Comment
    comment =  "Mean value of Digi " + it->second[1];
    trackerMap.setText(it->first, comment);
  }
  trackerMap.print(true);
  return;
}
//
// -- Browse through monitorable and get values need for TrackerMap
//
void SiStripActionExecutor::getValuesForTkMap(MonitorUserInterface* mui,
         std::string me_name, SiStripActionExecutor::DetMapType & values) {
  std::string currDir = mui->pwd();
  // browse through monitorable; check if MEs exist
  if (currDir.find("detector") != std::string::npos)  {
    TCanvas canvas("display");
    std::string status;
    std::vector<std::string> contents = mui->getMEs();    
    for (std::vector<std::string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      if ((*it).find(me_name) == 0) {
	std::string fullpathname = currDir + "/" + (*it); 
        MonitorElement * me = mui->get(fullpathname);
        if (me) {
          // Geometric ID
	  int id = atoi((currDir.substr(currDir.find("detector_")+9)).c_str());
          // Mean Value
	  ostringstream mean_str;
	  mean_str << me->getMean();
          // Status after comparison to Referece 
	  if (me->hasError()) status = "Error";
	  else if (me->hasWarning()) status = "Warning";
	  else  status = "Ok";
          // creation of jpg file
	  canvas.Clear();
	  // Access the Root object
	  MonitorElementT<TNamed>* ob = 
	    dynamic_cast<MonitorElementT<TNamed>*>(me);
	  if (ob) {
	    ob->operator->()->Draw();
	    ostringstream name_str;
	    name_str << id << ".jpg";
	    canvas.SaveAs(name_str.str().c_str());
	  }
          vector<std::string> vtemp;
          vtemp.push_back(status);
          vtemp.push_back(mean_str.str());  
	  values.insert(pair<int,std::vector <std::string> >(id, vtemp));
        }
      }
    }
  } else {
    std::vector<std::string> subdirs = mui->getSubdirs();
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      getValuesForTkMap(mui, me_name, values);
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
    std::string tag = "Summary" 
                        + me_name 
                        + currDir.substr(currDir.find(dir_name)+dir_name.size());
    MonitorElement* sum_me = getSummaryME(mui, tag);
    if (!sum_me) return;
    std::vector<std::string> subdirs = mui->getSubdirs();
    int ndet = 0;
    for (std::vector<std::string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      if ( (*it).find("detector") == std::string::npos) continue;
      mui->cd(*it);
      ndet++;
      std::vector<std::string> contents = mui->getMEs();    
      for (std::vector<std::string>::const_iterator im = contents.begin();
	 im != contents.end(); im++) {
	if ((*im).find("DigisPerDetector") == 0) {
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

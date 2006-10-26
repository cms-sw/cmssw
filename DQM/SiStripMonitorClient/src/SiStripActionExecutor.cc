#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "CommonTools/TrackerMap/interface/TrackerMap.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"
//#include "DQMServices/ClientConfig/interface/QTestHandle.h"
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
  
  getValuesForTkMap(mui, me_names, valueMap);  
  if (valueMap.size() == 0) return;
  MonitorElement* tkmap_me;
  mui->cd();
  string path = mui->pwd();
  path += "/TrackerMapSummary";
  tkmap_me = mui->get(path);

  if (tkmap_me) {
    MonitorElementT<TNamed>* obh1 = 
      dynamic_cast<MonitorElementT<TNamed>*> (tkmap_me);
    if (obh1) {
      TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
      if (root_obh1) root_obh1->Reset();        
    }  
  } else {
    DaqMonitorBEInterface * bei = mui->getBEInterface();
    tkmap_me = bei->book1D("TrackerMapSummary", "Summary Info for TrackerMap",
                  valueMap.size(),0.5,valueMap.size()+0.5);
  }
  int rval = 0;
  int gval = 0;
  int bval = 0;
  int ibin = 0;
  for (SiStripActionExecutor::DetMapType::const_iterator it = valueMap.begin();
       it != valueMap.end(); it++) {
    ibin++;
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
	 << " " << comment.str()
         << " Status : " << istat << endl;
    trackerMap.fillc(it->first, rval, gval, bval);
    tkmap_me->Fill(ibin, istat);
    ostringstream det_id;
    det_id << it->first;
    tkmap_me->setBinLabel(ibin, det_id.str());
    
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
  if (!configParser_->getMENamesForSummary(structure_name, summaryMENames)) {
    cout << "SiStripActionExecutor::createSummary: Failed to read Summary configuration parameters!! ";
    return;
  }
  mui->cd();
  fillSummary(mui);
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
void SiStripActionExecutor::fillSummary(MonitorUserInterface* mui) {
  string currDir = mui->pwd();
  vector<string> subdirs = mui->getSubdirs();
  int nmod = 0;
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    if ( (*it).find("module_") == string::npos) continue;
    nmod++;       
  }  
  if (nmod > 0) {
    fillSummaryHistos(mui);
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      mui->cd(*it);
      fillSummary(mui);
      mui->goUp();
    }
    fillGrandSummaryHistos(mui);
  }
}
void SiStripActionExecutor::fillGrandSummaryHistos(MonitorUserInterface* mui) {
  map<string, MonitorElement*> MEMap;
  string currDir = mui->pwd();
  string dir_name =  currDir.substr(currDir.find_last_of("/")+1);
  if ((dir_name.find("SiStrip") == 0) ||
      (dir_name.find("Collector") == 0) ||
      (dir_name.find("MechanicalView") == 0) ||
      (dir_name.find("FU0") == 0) ) return;
  vector<string> subdirs = mui->getSubdirs();
  if (subdirs.size() == 0) return;;
  int iDir =0;
  int nbin = 0; 
  for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
    mui->cd(*it);
    vector<string> contents = mui->getMEs();
    mui->goUp();
    
    for (vector<string>::const_iterator isum = summaryMENames.begin();
	   isum != summaryMENames.end(); isum++) {
	string name = "Summary_" + (*isum) + "_in_" 
                       + currDir.substr(currDir.find_last_of("/")+1);
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
	if ((*im).find((*isum)) != string::npos) {
	  string full_path = currDir + "/" + (*it) + "/" +(*im);
	  MonitorElement * me_i = mui->get(full_path);
	  if (!me_i) continue; 
          map<string, MonitorElement*>::iterator iPos = MEMap.find((*isum)); 
          MonitorElement* me; 
          if (iPos == MEMap.end()) {
	     MonitorElementT<TNamed>* obh1 = 
	       dynamic_cast<MonitorElementT<TNamed>*> (me_i);
	     if (obh1) {
	       TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	       if (root_obh1) nbin = root_obh1->GetNbinsX();        
	     } else nbin = 20;
	     me = getSummaryME(mui, name, nbin);
	     MEMap.insert(pair<string, MonitorElement*>(*isum, me));
          } else  me =  iPos->second;
	  for (int k = 1; k < nbin+1; k++) {
	    me->setBinContent(k+(iDir*nbin), me_i->getBinContent(k));
	  }
	}
      }
    }
    iDir++;
  }
}
//
// -- Get Summary ME
//
MonitorElement* SiStripActionExecutor::getSummaryME(MonitorUserInterface* mui, 
                         string& name, int nval) {
  MonitorElement* me = 0;
  DaqMonitorBEInterface * bei = mui->getBEInterface();
  int nDir = mui->getSubdirs().size();
  int nBins  = nval * nDir;
  string currDir = mui->pwd();
  // If already booked
  vector<string> contents = mui->getMEs();    
  for (vector<string>::const_iterator it = contents.begin();
       it != contents.end(); it++) {
    if ((*it).find(name) == 0) {
      string fullpathname = mui->pwd() + "/" + (*it); 
      me = mui->get(fullpathname);
      if (me) {
	MonitorElementT<TNamed>* obh1 = 
	  dynamic_cast<MonitorElementT<TNamed>*> (me);
	if (obh1) {
	  TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	  if (root_obh1) {
            root_obh1->Reset();
            root_obh1->GetXaxis()->LabelsOption("uv");
          }
	  break;
	}
      }
    }
  }
  if (!me) {
    me = bei->book1D(name, name,nBins,0.5,nBins+0.5);
    if (me) {
      MonitorElementT<TNamed>* obh1 = 
	dynamic_cast<MonitorElementT<TNamed>*> (me);
      if (obh1) {
	TH1F * root_obh1 = dynamic_cast<TH1F *> (obh1->operator->());
	if (root_obh1) {
	  root_obh1->GetXaxis()->LabelsOption("uv");
	}
      }
    }
    vector<string> subdirs = mui->getSubdirs();  
    int ibin = nval/2;
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      me->setBinLabel(ibin, (*it));
      ibin += nval;
    }
  }
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
  cout << " Setting Up Quality Tests " << endl;
  //  QTestHandle qtester;
  //  if(!qtester.configureTests("sistrip_qualitytest_config.xml", mui)){
  //    cout << " Attaching Qtests to MEs" << endl;
  //    qtester.attachTests(mui);			
  //  }
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
//
// -- Save Monitor Elements in a file
//      
void SiStripActionExecutor::saveMEs(MonitorUserInterface* mui, string fname){
    mui->save(fname);
}
void SiStripActionExecutor::fillSummaryHistos(MonitorUserInterface* mui) {
  string currDir = mui->pwd();
  map<string, MonitorElement*> MEMap;
  vector<string> subdirs = mui->getSubdirs();
  if (subdirs.size() ==0) return;
  
  int ndet = 0;
  int nval = 1;
  for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
    if ( (*it).find("module_") == string::npos) continue;
    mui->cd(*it);
    ndet++;
    vector<string> contents = mui->getMEs();    
    mui->goUp();
    for (vector<string>::const_iterator isum = summaryMENames.begin();
	   isum != summaryMENames.end(); isum++) {
      string name = "Summary_" + (*isum) + "_in_" 
                     + currDir.substr(currDir.find_last_of("/")+1);
      for (vector<string>::const_iterator im = contents.begin();
	   im != contents.end(); im++) {
        if ((*im).find(*isum) != string::npos) {
	  string full_path = mui->pwd() + "/" +(*it)+ "/" + (*im);
	  MonitorElement * me_i = mui->get(full_path);
          if (!me_i) continue;
          map<string, MonitorElement*>::iterator iPos = MEMap.find((*isum)); 
          MonitorElement* me;
          if (iPos == MEMap.end()) {
            if ((*isum).find("Noise") != string::npos)   {
	      MonitorElementT<TNamed>* obh1 = 
		dynamic_cast<MonitorElementT<TNamed>*> (me_i);
	      if (obh1) {
		TProfile* root_obh1 = dynamic_cast<TProfile *> (obh1->operator->());
		if (root_obh1) nval = root_obh1->GetNbinsX();
	      }
            }
            me = getSummaryME(mui, name, nval);
            MEMap.insert(pair<string, MonitorElement*>(*isum, me));
          } else  me =  iPos->second;
 
            if ((*isum).find("Noise") != string::npos) {
              for (int k=1; k<nval+1; k++) {
                me->setBinContent((k+(ndet-1)*nval),me_i->getBinContent(k));
              }
            } else me->Fill(ndet*1.0, me_i->getMean());
          break;
        }
      }
    }
  }
}

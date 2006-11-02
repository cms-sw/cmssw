#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripQualityTester.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMapCreator.h"
//#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"


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
  collationDone = false;
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
 
  // Create and Fill the Tracker Map
  mui->cd();
  if (collationDone) mui->cd("Collector/Collated/SiStrip");

  TrackerMapCreator tkmap_creator;
  tkmap_creator.create(mui, me_names);
  
  mui->cd();  
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
  if (collationDone) {
    cout << " Creating Summary with Collated Monitor Elements " << endl;
    mui->cd("Collector/Collated/SiStrip");
    fillSummary(mui);
    mui->cd();
} else fillSummary(mui);
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
	    TProfile* prof = ExtractTObject<TProfile>().extract(me_i);
	    TH1F* hist1 = ExtractTObject<TH1F>().extract(me_i);
            if (prof) nbin = prof->GetNbinsX();
            else if (hist1) nbin = hist1->GetNbinsX();
	    else nbin = 20;
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
	TH1F* hist1 = ExtractTObject<TH1F>().extract(me);
	if (hist1) {
          hist1->Reset();
          hist1->LabelsOption("uv");
	}
      }
      break;
    }
  }
  if (!me) {
    me = bei->book1D(name, name,nBins,0.5,nBins+0.5);
    vector<string> subdirs = mui->getSubdirs();  
    int ibin = nval/2 + 1;
    for (vector<string>::const_iterator it = subdirs.begin();
       it != subdirs.end(); it++) {
      me->setBinLabel(ibin, (*it));
      //      cout << " TEST " << ibin << " " << (*it) << endl;
      ibin += nval;
    }
  }
  return me;
}
//
// -- Setup Quality Tests 
//
void SiStripActionExecutor::setupQTests(MonitorUserInterface * mui) {
  SiStripQualityTester qtester;
  mui->cd();
  if (collationDone) mui->cd("Collector/Collated/SiStrip");
  qtester.setupQTests(mui);
  mui->cd();
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
  map<string, vector<string> > collation_map;
  vector<string> contentVec;
  mui->getContents(contentVec);

  for (vector<string>::iterator it = contentVec.begin();
      it != contentVec.end(); it++) {
    if ((*it).find("module_") == string::npos) continue;
    string dir_path;
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), dir_path, contents);
    string tag = dir_path.substr(dir_path.find("module_")+7, dir_path.size()-1);
    for (vector<string>::iterator ic = contents.begin(); ic != contents.end(); ic++) {
      
      string me_path = dir_path + (*ic);
      string path = dir_path.substr(dir_path.find("SiStrip"),dir_path.size());
      MonitorElement* me = mui->get( me_path );
      TProfile* prof = ExtractTObject<TProfile>().extract( me );
      TH1F* hist1 = ExtractTObject<TH1F>().extract( me );
      TH2F* hist2 = ExtractTObject<TH2F>().extract( me );
      CollateMonitorElement* coll_me = 0;
      string coll_dir = "Collector/Collated/"+path;
      map<string, vector<string> >::iterator ipos = collation_map.find(tag);
      if(ipos == collation_map.end()) {
        if (collation_map[tag].capacity() != contents.size()) { 
          collation_map[tag].reserve(contents.size()); 
        }
        if      (hist1) coll_me = mui->collate1D((*ic),(*ic),coll_dir);
        else if (hist2) coll_me = mui->collate2D((*ic),(*ic),coll_dir);
        else if (prof) coll_me = mui->collate2D((*ic),(*ic),coll_dir);
        collation_map[tag].push_back(coll_dir+(*ic));
      } else {
        if (find(ipos->second.begin(), ipos->second.end(), (*ic)) == ipos->second.end()){
	  if (hist1)      coll_me = mui->collate1D((*ic),(*ic),coll_dir);
	  else if (hist2) coll_me = mui->collate2D((*ic),(*ic),coll_dir);
	  else if (prof)  coll_me = mui->collateProf((*ic),(*ic),coll_dir);
	  collation_map[tag].push_back(coll_dir+(*ic));	  
        }
      }
      if (coll_me) mui->add(coll_me, me_path);
    }
  }
  collationDone = true;
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

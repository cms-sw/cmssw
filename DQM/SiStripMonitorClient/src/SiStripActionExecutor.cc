#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMapCreator.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiStripActionExecutor::SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Creating SiStripActionExecutor " << "\n" ;
  configParser_ = 0;
  qtHandler_ = 0;
  summaryCreator_= 0;
  collationDone = false;
}
//
// --  Destructor
// 
SiStripActionExecutor::~SiStripActionExecutor() {
  edm::LogInfo("SiStripActionExecutor") << 
    " Deleting SiStripActionExecutor " << "\n" ;
  if (configParser_) delete configParser_;
  if (qtHandler_) delete qtHandler_;
  if (summaryCreator_) delete   summaryCreator_;

}
//
// -- Read Configurationn File
//
void SiStripActionExecutor::readConfiguration() {
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiStripConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
}
//
// -- Read Configurationn File
//
bool SiStripActionExecutor::readConfiguration(int& tkmap_freq, int& sum_freq) {
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_monitorelement_config.xml");
  if (configParser_ == 0) {
    configParser_ = new SiStripConfigParser();
    configParser_->getDocument(edm::FileInPath(localPath).fullPath());
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
  if (!configParser_->getMENamesForTrackerMap(tkmap_name, tkMapMENames)){
    cout << "SiStripActionExecutor::createTkMap: Failed to read TrackerMap configuration parameters!! ";
    return;
  }
  cout << " # of MEs in Tk Map " << tkMapMENames.size() << endl;
 
  // Create and Fill the Tracker Map
  mui->cd();
  if (collationDone) mui->cd("Collector/Collated/SiStrip");

  TrackerMapCreator tkmap_creator;
  tkmap_creator.create(mui, tkMapMENames);
  
  mui->cd();  
}
// -- Browse through the Folder Structure
//
void SiStripActionExecutor::createSummary(MonitorUserInterface* mui) {
  if (!summaryCreator_) summaryCreator_ = new SiStripSummaryCreator();

  map<string,string> summary_mes;
  if (!configParser_->getMENamesForSummary(summary_mes)) {
    cout << "SiStripActionExecutor::createSummary: Failed to read Summary configuration parameters!! ";
    return;
  }
  summaryCreator_->setSummaryMENames(summary_mes);
  mui->cd();
  if (collationDone) {
    cout << " Creating Summary with Collated Monitor Elements " << endl;
    mui->cd("Collector/Collated/SiStrip");
    summaryCreator_->createSummary(mui);
    mui->cd();
  } else summaryCreator_->createSummary(mui);
}
//
// -- Setup Quality Tests 
//
void SiStripActionExecutor::setupQTests(MonitorUserInterface * mui) {
  mui->cd();
  if (collationDone) mui->cd("Collector/Collated/SiStrip");
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_qualitytest_config.xml");
  if (!qtHandler_) {
    qtHandler_ = new QTestHandle();
  }
  if(!qtHandler_->configureTests(edm::FileInPath(localPath).fullPath(),mui)){
    cout << " Setting Up Quality Tests " << endl;
    qtHandler_->attachTests(mui);			
    mui->cd();
  } else {
    cout << " Problem to Set Up Quality Tests " << endl;
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
//
// -- Save Monitor Elements in a file
//      
void SiStripActionExecutor::saveMEs(MonitorUserInterface* mui, string fname){
  if (collationDone) {
    mui->save(fname,"Collector/Collated");
  } else {
     mui->save(fname,mui->pwd(),90);
  }
}
//
// -- Get TkMap ME names
//
int SiStripActionExecutor::getTkMapMENames(std::vector<std::string>& names) {
  if (tkMapMENames.size() == 0) return 0;
  for (vector<string>::iterator it = tkMapMENames.begin();
       it != tkMapMENames.end(); it++) {
    names.push_back(*it) ;
  }
  return names.size();
}

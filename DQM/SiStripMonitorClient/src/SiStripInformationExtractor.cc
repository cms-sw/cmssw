
#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/QReport.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripHistoPlotter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"


#include <iostream>
using namespace std;

//
// -- Constructor
// 
SiStripInformationExtractor::SiStripInformationExtractor() {
  edm::LogInfo("SiStripInformationExtractor") << 
    " Creating SiStripInformationExtractor " << "\n" ;
  layoutParser_ = 0;
  layoutMap.clear();
  histoPlotter_=0;
  histoPlotter_ = new SiStripHistoPlotter();
  readConfiguration();
}
//
// --  Destructor
// 
SiStripInformationExtractor::~SiStripInformationExtractor() {
  edm::LogInfo("SiStripInformationExtractor") << 
    " Deleting SiStripInformationExtractor " << "\n" ;
  if (layoutParser_) delete layoutParser_;
  if (histoPlotter_) delete histoPlotter_;

}
//
// -- Read Configurationn File
//
void SiStripInformationExtractor::readConfiguration() {
  string localPath = string("DQM/SiStripMonitorClient/data/sistrip_plot_layout.xml");
  if (layoutParser_ == 0) {
    layoutParser_ = new SiStripLayoutParser();
    layoutParser_->getDocument(edm::FileInPath(localPath).fullPath());
  }
  if (layoutParser_->getAllLayouts(layoutMap)) {
     edm::LogInfo("SiStripInformationExtractor") << 
    " Layouts correctly readout " << "\n" ;
  } else  edm::LogInfo("SiStripInformationExtractor") << 
          " Problem in reading Layout " << "\n" ;
  if (layoutParser_) delete layoutParser_;

}
//
// --  Fill Summary Histo List
// 
void SiStripInformationExtractor::printSummaryHistoList(DQMStore * dqm_store, ostringstream& str_val){
  static string indent_str = "";

  string currDir = dqm_store->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  if (dname.find("module_") ==0) return;
  str_val << "<li><a href=\"#\" id=\"" 
          << currDir << "\">" << dname << "</a>" << endl;
  vector<MonitorElement *> meVec = dqm_store->getContents(currDir);
  vector<string> subDirVec = dqm_store->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;      
  for (vector<MonitorElement *>::const_iterator it = meVec.begin();
         it != meVec.end(); it++) {
    MonitorElement* me = (*it);
    if (!me) continue;
    string name = (*it)->getName();
    //    if (name.find("Summary") == 0) {
      str_val << "<li class=\"dhtmlgoodies_sheet.gif\">"
              << " <a href=\"javascript:RequestHistos.DrawSingleHisto('"
              << currDir 
              << "')\">" << name 
              << "</a></li>" << endl;
      //    }
  }

  string mtag ="Modules: ";  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    string titl = (*ic);
    if (titl.find("module_") == 0)  {
      titl = titl.substr(titl.find("module_")+7);
      mtag += titl + " ";
    }
    printSummaryHistoList(dqm_store, str_val);
    dqm_store->goUp();
  }
  if (mtag.size() > 10) {
    str_val << "<li class=\"note.gif\"><a href=\"#\">" << mtag << "</a></li>" << endl;
  }
  str_val << "</ul> "<< endl;  
  str_val << "</li> "<< endl;  
}
//
// --  Fill Alarm List
// 
void SiStripInformationExtractor::printAlarmList(DQMStore * dqm_store, ostringstream& str_val){
  static string indent_str = "";

  string currDir = dqm_store->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  string image_name;
  selectImage(image_name,dqm_store->getStatus(currDir));
  str_val << "<li><a href=\"#\" id=\"" 
          << currDir << "\">" << dname << "</a> <img src=\"" 
          << image_name << "\"></img>" << endl;
  vector<string> subDirVec = dqm_store->getSubdirs();
  //  vector<string> meVec = dqm_store->getMEs(); 
  vector<MonitorElement *> meVec = dqm_store->getContents(currDir);
  
  if (subDirVec.size() == 0 && meVec.size() == 0) {
    str_val << "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;
  if (dname.find("module_") != string::npos) {
    if (meVec.size() > 0) {
      for (vector<MonitorElement *>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
        MonitorElement * me = (*it);
	if (!me) continue;
        vector<QReport*> q_reports = me->getQReports();
        if (q_reports.size() > 0) {
	  string image_name1;
	  selectImage(image_name1,q_reports);
	  str_val << "<li class=\"dhtmlgoodies_sheet.gif\">"
		  << " <a href=\"javascript:RequestHistos.ReadStatus('"
		  << currDir
		  << "')\">" << me->getName()
		  << "</a><img src=\"" << image_name1 << "\"></img>"
	  	  << " </li>" << endl;
        }
      }
    }
  }
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    printAlarmList(dqm_store, str_val);
    dqm_store->goUp();
  }
  str_val << "</ul> "<< endl;  
  str_val << "</li> "<< endl;  
}
//
// -- Select Histograms for a given module
//
void SiStripInformationExtractor::getSingleModuleHistos(DQMStore * dqm_store, 
      const multimap<string, string>& req_map, xgi::Output * out){

  vector<string> hlist;
  getItemList(req_map,"histo", hlist);

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";
  
  SiStripFolderOrganizer folder_organizer;
  string path;
  folder_organizer.getFolderName(detId,path);   

  vector<MonitorElement*> all_mes = dqm_store->getContents(path);
  setHTMLHeader(out);
  *out << path << " ";

  for (vector<string>::const_iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      string hname = me->getName();
      string name = hname.substr(0, hname.find("__det__"));
      if (name == (*ih)) {
	string full_path = path + "/" + hname;
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << hname << " " ;
      }
    }
  }
}
//
// -- Select Histograms from Global folder
//
void SiStripInformationExtractor::getGlobalHistos(DQMStore* dqm_store, const multimap<string, string>& req_map, xgi::Output * out) {
 
  vector<string> hlist;  
  getItemList(req_map,"histo", hlist);

  string path = getItemValue(req_map, "GlobalFolder");    

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";

  vector<MonitorElement *> all_mes = dqm_store->getContents(path);

  setHTMLHeader(out);
  *out << path << " ";

  for (vector<string>::const_iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {      
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      string hname = me->getName();
      string name = hname.substr(0, hname.find("__det__"));
      if (name == (*ih)) {
	string full_path = path + "/" + hname;
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << hname << " " ;
      }
    }
  }
}
//
// -- Get All histograms from a Path
//
void SiStripInformationExtractor::getHistosFromPath(DQMStore * dqm_store, const std::multimap<std::string, std::string>& req_map, xgi::Output * out){

  string path = getItemValue(req_map,"Path");

  if (path.size() == 0) return;

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";

  setHTMLHeader(out);
  vector<MonitorElement*> all_mes = dqm_store->getContents(path);
  *out << path << " " ;
  for(vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
    MonitorElement* me = (*it);
    if (!me) continue;
    string name = me->getName();
    string full_path = path + "/" + name;

    histoPlotter_->setNewPlot(full_path, opt, width, height);
    *out << name << " ";
  }
}
//
// plot Histograms from Layout
//
void SiStripInformationExtractor::plotHistosFromLayout(DQMStore * dqm_store){
  if (layoutMap.size() == 0) return;

  ofstream image_file;
  ofstream title_file;
  
  for (map<std::string, std::vector< std::string > >::iterator it = layoutMap.begin() ; it != layoutMap.end(); it++) {
    int ival = 0;
    string image_list = "images/" + it->first +".lis";
    image_file.open(image_list.c_str(), ios::out);
    if (!image_file) return;

    string title_list = "images/" + it->first +"_titles.lis";
    title_file.open(title_list.c_str(), ios::out);
    if (!title_file) return;

    image_file << "[";
    title_file << "[";
    for (vector<string>::iterator im = it->second.begin(); 
	 im != it->second.end(); im++) {  
      string path_name = (*im);
      if (path_name.size() == 0) continue;
      MonitorElement* me = dqm_store->get(path_name);
      ival++;
      ostringstream  fname, ftitle;
      if (!me) {
        fname << "images/EmptyPlot.png";
        ftitle << "EmptyPlot";
        
      } else {
	fname << "images/" << it->first << "_" <<ival << ".png";
        ftitle << me->getName();
        histoPlotter_->createStaticPlot(me, fname.str());
      }
      if (ival != it->second.size()) {
        image_file << "\"" << fname.str() << "\","<< endl;
	title_file << "\"" << ftitle.str() << "\","<< endl;
      } else {
        image_file << "\"" << fname.str() << "\"" << endl;
	title_file << "\"" << ftitle.str() << "\""<< endl;
      }
    }
    image_file << "]" << endl;
    image_file.close();
    title_file << "]" << endl;
    title_file.close();
  }
}
//
// -- Plot Tracker Map MEs
//
void SiStripInformationExtractor::getTrackerMapHistos(DQMStore* dqm_store, const std::multimap<std::string, std::string>& req_map, xgi::Output * out) {

  vector<string> hlist;
  string tkmap_name;
  SiStripConfigParser config_parser;
  string localPath = string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  config_parser.getDocument(edm::FileInPath(localPath).fullPath());
  if (!config_parser.getMENamesForTrackerMap(tkmap_name, hlist)) return;
  if (hlist.size() == 0) return;

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";
  
  SiStripFolderOrganizer folder_organizer;
  string path;
  folder_organizer.getFolderName(detId,path);   

  vector<MonitorElement*> all_mes = dqm_store->getContents(path);
  setHTMLHeader(out);
  *out << path << " ";
  for (vector<string>::iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
    for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      string hname = me->getName(); 
      string name = hname.substr(0, hname.find("__det__"));
      if (name == (*ih)) {	
	string full_path = path + "/" + hname;
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << hname << " " ;
      }      
    }
  }   
}

//
// -- Get a tagged image 
//
void SiStripInformationExtractor::getIMGCImage(const multimap<string, string>& req_map, xgi::Output * out){
  
  string path = getItemValue(req_map,"Path");
  string image;
  histoPlotter_->getNamedImageBuffer(path, image);

  out->getHTTPResponseHeader().addHeader("Content-Type", "image/png");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << image;

}
//
// -- Read Layout Group names
//
void SiStripInformationExtractor::readLayoutNames(multimap<string, string>& req_map, xgi::Output * out){
  if (layoutMap.size() > 0) {
    setXMLHeader(out);
    *out << "<LayoutList>" << endl;
   for (map<string, vector< string > >::iterator it = layoutMap.begin();
	it != layoutMap.end(); it++) {
     *out << "<LName>" << it->first << "</LName>" << endl;  
   }
   *out << "</LayoutList>" << endl;
  }  
}
//
// read the Module And HistoList
//
void SiStripInformationExtractor::readModuleAndHistoList(DQMStore* dqm_store, const edm::ESHandle<SiStripDetCabling>& detcabling, xgi::Output * out) {

  std::vector<uint32_t> SelectedDetIds;
  detcabling->addActiveDetectorsRawIds(SelectedDetIds);

  setXMLHeader(out);
  *out << "<ModuleAndHistoList>" << endl;


  // Fill Module List
  *out << "<ModuleList>" << endl;
  uint32_t aDetId  = 0;
  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){    
    uint32_t detId = *idetid;
    if (detId == 0 || detId == 0xFFFFFFFF) continue;
    if (aDetId == 0) aDetId = detId;
    ostringstream detIdStr;
    detIdStr << detId;
    *out << "<ModuleNum>" << detIdStr.str() << "</ModuleNum>" << endl;     
  }
  *out << "</ModuleList>" << endl;
  // Fill Histo list
  *out << "<HistoList>" << endl;

  SiStripFolderOrganizer folder_organizer;
  string dir_path;
  folder_organizer.getFolderName(aDetId,dir_path);     
  vector<MonitorElement*> detector_mes = dqm_store->getContents(dir_path);
  for (vector<MonitorElement *>::const_iterator it = detector_mes.begin();
       it!= detector_mes.end(); it++) {
    MonitorElement * me = (*it);     
    if (!me) continue;
    string hname_full = me->getName();
    string hname = hname_full.substr(0, hname_full.find("__det__"));
    *out << "<Histo>" << hname << "</Histo>" << endl;     
  }
  *out << "</HistoList>" << endl;
  *out << "</ModuleAndHistoList>" << endl;

  /*   std::vector<std::string> hnames;
   std::vector<std::string> mod_names;
   fillModuleAndHistoList(dqm_store, mod_names, hnames);
   setXMLHeader(out);
  *out << "<ModuleAndHistoList>" << endl;
  *out << "<ModuleList>" << endl;
   for (std::vector<std::string>::iterator im = mod_names.begin();
        im != mod_names.end(); im++) {
     *out << "<ModuleNum>" << *im << "</ModuleNum>" << endl;     
   }
   *out << "</ModuleList>" << endl;
   *out << "<HistoList>" << endl;

   for (std::vector<std::string>::iterator ih = hnames.begin();
        ih != hnames.end(); ih++) {
     *out << "<Histo>" << *ih << "</Histo>" << endl;     

   }
   *out << "</HistoList>" << endl;
   *out << "</ModuleAndHistoList>" << endl;
*/
}
//
// Global Histogram List
//
void SiStripInformationExtractor::readGlobalHistoList(DQMStore* dqm_store, std::string& str_name,xgi::Output * out) {
   std::vector<std::string> hnames;
   string dname = str_name;
  
   setXMLHeader(out);
   *out << "<GlobalHistoList>" << endl;
   if (dqm_store->dirExists(dname)) {
     vector<MonitorElement*> meVec = dqm_store->getContents(dname);
     for (vector<MonitorElement *>::const_iterator it = meVec.begin();
	  it != meVec.end(); it++) {
       MonitorElement* me = (*it);
       if (!me) continue;
       *out << "<GHisto>" << (*it)->getName() << "</GHisto>" << endl;           
     }
   } else {   
     *out << "<GHisto>" << " Desired directory : " << "</GHisto>" << endl;
     *out << "<GHisto>" <<       dname             << "</GHisto>" << endl;
     *out << "<GHisto>" << " does not exist!!!!  " << "</GHisto>" << endl;      
   }
   *out << "</GlobalHistoList>" << endl;
}
//
// read the Structure And SummaryHistogram List
//
void SiStripInformationExtractor::readSummaryHistoTree(DQMStore* dqm_store, string& str_name, xgi::Output * out) {
  ostringstream sumtree;
  string dname = "SiStrip/" + str_name;
  if (dqm_store->dirExists(dname)) {    
    dqm_store->cd(dname);
    sumtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printSummaryHistoList(dqm_store,sumtree);
    sumtree <<"</ul>" << endl;   
  } else {
    sumtree << " Desired Directory :  " << endl;
    sumtree <<       dname              << endl;
    sumtree <<  " does not exist !!!! " << endl;
  }
  setPlainHeader(out);
  *out << sumtree.str();
   dqm_store->cd();
}
//
// read the Structure And Alarm Tree
//
void SiStripInformationExtractor::readAlarmTree(DQMStore* dqm_store, 
                  string& str_name, xgi::Output * out){
  ostringstream alarmtree;
  string dname = "SiStrip/" + str_name;
  if (dqm_store->dirExists(dname)) {    
    dqm_store->cd(dname);
    alarmtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printAlarmList(dqm_store,alarmtree);
    alarmtree <<"</ul>" << endl; 
  } else {
    alarmtree << "Desired Directory :   " << endl;
    alarmtree <<       dname              << endl;
    alarmtree <<  " does not exist !!!! " << endl;
  }
  setPlainHeader(out);
  *out << alarmtree.str();
   dqm_store->cd();
}
//
// Get elements from multi map
//
void SiStripInformationExtractor::getItemList(const multimap<string, string>& req_map, string item_name,vector<string>& items) {
  items.clear();
  for (multimap<string, string>::const_iterator it = req_map.begin();
       it != req_map.end(); it++) {
    
    if (it->first == item_name) {
      items.push_back(it->second);
    }
  }
}
//
//  check a specific item in the map
//
bool SiStripInformationExtractor::hasItem(multimap<string,string>& req_map,
					  string item_name){
  multimap<string,string>::iterator pos = req_map.find(item_name);
  if (pos != req_map.end()) return true;
  return false;  
}
//
// check the value of an item in the map
//  
string SiStripInformationExtractor::getItemValue(const multimap<string,string>& req_map,
						 std::string item_name){
  multimap<string,string>::const_iterator pos = req_map.find(item_name);
  string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
  return value;
}
//
// -- Get color  name from status
//
void SiStripInformationExtractor::selectColor(string& col, int status){
  if (status == dqm::qstatus::STATUS_OK)    col = "#00ff00";
  else if (status == dqm::qstatus::WARNING) col = "#ffff00";
  else if (status == dqm::qstatus::ERROR)   col = "#ff0000";
  else if (status == dqm::qstatus::OTHER)   col = "#ffa500";
  else  col = "#0000ff";
}
//
// -- Get Image name from ME
//
void SiStripInformationExtractor::selectColor(string& col, vector<QReport*>& reports){
  int istat = 999;
  int status = 0;
  for (vector<QReport*>::const_iterator it = reports.begin(); it != reports.end();
       it++) {
    status = (*it)->getStatus();
    if (status > istat) istat = status;
  }
  selectColor(col, status);
}
//
// -- Get Image name from status
//
void SiStripInformationExtractor::selectImage(string& name, int status){
  if (status == dqm::qstatus::STATUS_OK) name="images/LI_green.gif";
  else if (status == dqm::qstatus::WARNING) name="images/LI_yellow.gif";
  else if (status == dqm::qstatus::ERROR) name="images/LI_red.gif";
  else if (status == dqm::qstatus::OTHER) name="images/LI_orange.gif";
  else  name="images/LI_blue.gif";
}
//
// -- Get Image name from ME
//
void SiStripInformationExtractor::selectImage(string& name, vector<QReport*>& reports){
  int istat = 999;
  int status = 0;
  for (vector<QReport*>::const_iterator it = reports.begin(); it != reports.end();
       it++) {
    status = (*it)->getStatus();
    if (status > istat) istat = status;
  }
  selectImage(name, status);
}
//
// -- Get Warning/Error Messages
//
void SiStripInformationExtractor::readStatusMessage(DQMStore* dqm_store, std::multimap<std::string, std::string>& req_map, xgi::Output * out){

  string path = getItemValue(req_map,"Path");

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  string opt =" ";

  ostringstream test_status;
  
  setXMLHeader(out);
  *out << "<StatusAndPath>" << endl;
  *out << "<PathList>" << endl;
  if (path.size() == 0) {
    *out << "<HPath>" << "NONE" << "</HPath>" << endl;     
    test_status << " ME Does not exist ! " << endl;
  } else {
    vector<MonitorElement*> all_mes = dqm_store->getContents(path);
    *out << "<HPath>" << path << "</HPath>" << endl;     
    for(vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
      MonitorElement* me = (*it);
      if (!me) continue;
      string name = me->getName();  

      vector<QReport*> q_reports = me->getQReports();
      if (q_reports.size() == 0 && name.find("StripQualityFromCondDB") == string::npos) continue;
      string full_path = path + "/" + name;
      histoPlotter_->setNewPlot(full_path, opt, width, height);

      if (q_reports.size() != 0) {
        test_status << " QTest Status for " << name << " : " << endl;
        test_status << " ======================================================== " << endl; 
        for (vector<QReport*>::const_iterator it = q_reports.begin(); it != q_reports.end();
	     it++) {
	  int status = (*it)->getStatus();
	  if (status == dqm::qstatus::WARNING) test_status << " Warning ";
	  else if (status == dqm::qstatus::ERROR) test_status << " Error  ";
	  else if (status == dqm::qstatus::STATUS_OK) test_status << " Ok  ";
	  else if (status == dqm::qstatus::OTHER) test_status << " Other(" << status << ") ";
	  string mess_str = (*it)->getMessage();
	  test_status <<  "&lt;br/&gt;";
	  mess_str = mess_str.substr(mess_str.find(" Test")+5);
	  test_status <<  " QTest Name  : " << mess_str.substr(0, mess_str.find(")")+1) << endl;
	  test_status << "&lt;br/&gt;";
	  test_status <<  " QTest Detail  : " << mess_str.substr(mess_str.find(")")+2) << endl;
	} 
	test_status << " ======================================================== " << endl;
      }
      *out << "<HPath>" << name << "</HPath>" << endl;         
    }    
  }
  *out << "</PathList>" << endl;
  *out << "<StatusList>" << endl;
  *out << "<Status>" << test_status.str() << "</Status>" << endl;      
  *out << "</StatusList>" << endl;
  *out << "</StatusAndPath>" << endl;
}
//
// -- Read the text Summary of QTest result
//
void SiStripInformationExtractor::readQTestSummary(DQMStore* dqm_store, string type, const edm::ESHandle<SiStripDetCabling>& detcabling, xgi::Output * out) {
  std::vector<uint32_t> SelectedDetIds;
  detcabling->addActiveDetectorsRawIds(SelectedDetIds);

  int nDetsWithError = 0;
  int nDetsWithWarning = 0;
  int nTotalError = 0;
  int nTotalWarning = 0;
  int nDetsTotal = 0;
  ostringstream qtest_summary, lite_summary;
  
  SiStripFolderOrganizer folder_organizer;
  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){    
    uint32_t detId = *idetid;
    if (detId == 0 || detId == 0xFFFFFFFF){
      edm::LogError("SiStripInformationExtractor") 
                  <<"SiStripInformationExtractor::readQTestSummary: " 
                  << "Wrong DetId !!!!!! " <<  detId << " Neglecting !!!!!! ";
      continue;
    }
    nDetsTotal++;
    string dir_path;
    folder_organizer.getFolderName(detId, dir_path);     
    vector<MonitorElement*> detector_mes = dqm_store->getContents(dir_path);
    int error_me = 0;
    int warning_me = 0;
    for (vector<MonitorElement *>::const_iterator it = detector_mes.begin();
	 it!= detector_mes.end(); it++) {
      MonitorElement * me = (*it);     
      if (!me) continue;
      vector<QReport*> q_reports = me->getQReports();
      if (!me->hasError() && !me->hasWarning() ) continue;
      if (q_reports.size() == 0) continue;
      if (me->hasError()) error_me++;
      if (me->hasWarning()) warning_me++;
      if (error_me == 1 || warning_me == 1) {
	qtest_summary << " Module = " << me->getPathname() << endl;
	qtest_summary << "====================================================================="<< endl; 
      }
      qtest_summary << me->getName() << endl; 
      for (vector<QReport*>::const_iterator it = q_reports.begin(); it != q_reports.end();
	   it++) {
	int status =  (*it)->getStatus();
	string mess_str = (*it)->getMessage();
        
	if (status == dqm::qstatus::STATUS_OK || status == dqm::qstatus::OTHER) continue;
	if (status == dqm::qstatus::ERROR)        qtest_summary << " ERROR =>   ";
        else if (status == dqm::qstatus::WARNING) qtest_summary << " WARNING => ";
	qtest_summary << mess_str.substr(0, mess_str.find(")")+1) 
                    << " Result  : "  << mess_str.substr(mess_str.find(")")+2) << endl;
      } 
    }
    if (error_me > 0)   {
      nDetsWithError++;
      nTotalError += error_me;
    }
    if (warning_me > 0) {
      nDetsWithWarning++;
      nTotalWarning += warning_me;
    }
  }
  lite_summary << " Total Detectors " << nDetsTotal << endl;
  lite_summary << " # of Detectors with Warning " << nDetsWithWarning << endl;
  lite_summary << " # of Detectors with Error " << nDetsWithError << endl;
  lite_summary << endl;
  lite_summary << endl;
  lite_summary << " Total # MEs with Warning " << nTotalWarning << endl;
  lite_summary << " Total # MEs with Error "   << nTotalError << endl;


  setPlainHeader(out);
  if (type == "Lite") *out << lite_summary.str();
  else {
   if (nDetsWithWarning == 0 && nDetsWithError ==0)  *out << lite_summary.str();
   else  *out << qtest_summary.str();
  }

  //  dqm_store->cd();
}
//
// -- Create Images 
//
void SiStripInformationExtractor::createImages(DQMStore* dqm_store){
  histoPlotter_->createPlots(dqm_store);
}

//
// -- Set HTML Header in xgi output
//
void SiStripInformationExtractor::setHTMLHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
}
//
// -- Set XML Header in xgi output
//
void SiStripInformationExtractor::setXMLHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;

}
//
// -- Set Plain Header in xgi output
//
void SiStripInformationExtractor::setPlainHeader(xgi::Output * out) {
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");

}
//
// read the Structure And Readout/Control Histogram List
//
void SiStripInformationExtractor::readNonGeomHistoTree(DQMStore* dqm_store, string& fld_name, xgi::Output * out) {
  ostringstream sumtree;
  string dname = "SiStrip/" + fld_name;
  if (dqm_store->dirExists(dname)) {    
    dqm_store->cd(dname);
    sumtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printNonGeomHistoList(dqm_store,sumtree);
    sumtree <<"</ul>" << endl;   
  } else {
    sumtree << " Desired Directory :  " << endl;
    sumtree <<       dname              << endl;
    sumtree <<  " does not exist !!!! " << endl;
  }
  cout << sumtree.str() << endl;
  setPlainHeader(out);
  *out << sumtree.str();
   dqm_store->cd();
}
//
// --  Fill Readout/Control Histo List
// 
void SiStripInformationExtractor::printNonGeomHistoList(DQMStore * dqm_store, ostringstream& str_val){
  static string indent_str = "";

  string currDir = dqm_store->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  str_val << "<li><a href=\"#\" id=\"" 
          << currDir << "\">" << dname << "</a>" << endl;
  vector<MonitorElement *> meVec = dqm_store->getContents(currDir);
  vector<string> subDirVec = dqm_store->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;      
  for (vector<MonitorElement *>::const_iterator it = meVec.begin();
         it != meVec.end(); it++) {
    MonitorElement* me = (*it);
    if (!me) continue;
    string name = (*it)->getName();
      str_val << "<li class=\"dhtmlgoodies_sheet.gif\">"
              << " <a href=\"javascript:RequestHistos.DrawSingleHisto('"
              << currDir 
              << "')\">" << name 
              << "</a></li>" << endl;
  }
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    printNonGeomHistoList(dqm_store, str_val);
    dqm_store->goUp();
  }
  str_val << "</ul> "<< endl;  
  str_val << "</li> "<< endl;  
}

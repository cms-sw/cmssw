#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/QReport.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripHistoPlotter.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"


#include <iostream>

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
  std::string localPath = std::string("DQM/SiStripMonitorClient/data/sistrip_plot_layout.xml");
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

  subdetVec.push_back("SiStrip/MechanicalView/TIB");
  subdetVec.push_back("SiStrip/MechanicalView/TOB");
  subdetVec.push_back("SiStrip/MechanicalView/TID/side_2");
  subdetVec.push_back("SiStrip/MechanicalView/TID/side_1");
  subdetVec.push_back("SiStrip/MechanicalView/TEC/side_2");
  subdetVec.push_back("SiStrip/MechanicalView/TEC/side_1");

}
//
// --  Fill Summary Histo List
// 
void SiStripInformationExtractor::printSummaryHistoList(DQMStore * dqm_store, std::ostringstream& str_val){
  static std::string indent_str = "";

  std::string currDir = dqm_store->pwd();
  std::string dname = currDir.substr(currDir.find_last_of("/")+1);
  if (dname.find("module_") ==0) return;
  //  str_val << "<li><a href=\"#\" id=\"" << currDir << "\">" << dname << "</a>" << std::endl;
  str_val << "<li><span class=\"folder\">" << dname << "</span>" << std::endl;
  std::vector<MonitorElement *> meVec = dqm_store->getContents(currDir);
  std::vector<std::string> subDirVec = dqm_store->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << "</li> "<< std::endl;    
    return;
  }
  str_val << "<ul>" << std::endl;      
  for (std::vector<MonitorElement *>::const_iterator it = meVec.begin();
         it != meVec.end(); it++) {
    MonitorElement* me = (*it);
    if (!me) continue;
    std::string name = (*it)->getName();
    str_val << "<li> <span class=\"file\"><a href=\"javascript:RequestHistos.DrawSummaryHistogram('" 
            << currDir
	    << "')\">" << name << "</a></span></li>" << std::endl;
  }

  std::string mtag ="Modules: ";  
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    std::string titl = (*ic);
    if (titl.find("module_") == 0)  {
      titl = titl.substr(titl.find("module_")+7);
      mtag += titl + " ";
    }
    printSummaryHistoList(dqm_store, str_val);
    dqm_store->goUp();
  }
  if (mtag.size() > 10) {
    //    str_val << "<li class=\"note.gif\"><a href=\"#\">" << mtag << "</a></li>" << std::endl;
    str_val << "<li> <span class=\"file\"><a href=\"#\">" << mtag << "</a></span></li>" << std::endl;
  }
  str_val << "</ul> "<< std::endl;  
  str_val << "</li> "<< std::endl;  
}
//
// --  Fill Alarm List
// 
void SiStripInformationExtractor::printAlarmList(DQMStore * dqm_store, std::ostringstream& str_val){
  static std::string indent_str = "";

  std::string currDir = dqm_store->pwd();
  std::string dname = currDir.substr(currDir.find_last_of("/")+1);
  std::string image_name;
  selectImage(image_name,dqm_store->getStatus(currDir));
  str_val << "<li><span class=\"folder\">" 
          << dname << "<img src=\""
          << image_name << "\"></span>" << std::endl;
  std::vector<std::string> subDirVec = dqm_store->getSubdirs();
  std::vector<MonitorElement *> meVec = dqm_store->getContents(currDir);
  
  if (subDirVec.size() == 0 && meVec.size() == 0) {
    str_val << "</li> "<< std::endl;    
    return;
  }
  str_val << "<ul>" << std::endl;
  if (dname.find("module_") != std::string::npos) {
    if (meVec.size() > 0) {
      for (std::vector<MonitorElement *>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
        MonitorElement * me = (*it);
	if (!me) continue;
        std::vector<QReport*> q_reports = me->getQReports();
        if (q_reports.size() > 0) {
	  std::string image_name1;
	  selectImage(image_name1,q_reports);
	  str_val << "<li><span class=\"file\"><a href=\"javascript:RequestHistos.ReadAlarmStatus('"
                  << currDir << "')\">"<<me->getName()
                  << "</a><img src=\""
                  << image_name1 << "\">" 
                  << "</span></li>" << std::endl;
        }
      }
    }
  }
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    printAlarmList(dqm_store, str_val);
    dqm_store->goUp();
  }
  str_val << "</ul> "<< std::endl;  
  str_val << "</li> "<< std::endl;  
}
//
// -- Select Histograms for a given module
//
void SiStripInformationExtractor::getSingleModuleHistos(DQMStore * dqm_store, 
    const std::multimap<std::string, std::string>& req_map, xgi::Output * out){

  std::vector<std::string> hlist;
  getItemList(req_map,"histo", hlist);

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  std::string opt =" ";
  
  SiStripFolderOrganizer folder_organizer;
  std::string path;
  folder_organizer.getFolderName(detId,path);   

  std::vector<MonitorElement*> all_mes = dqm_store->getContents(path);
  setXMLHeader(out);
  *out << "<HPathAndHNameList>" << std::endl;
  *out << "<HPath>" << path << "</HPath>" << std::endl;

  for (std::vector<std::string>::const_iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {
    for (std::vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      std::string hname = me->getName();
      std::string name = hname.substr(0, hname.find("__det__"));
      if (name == (*ih)) {
	std::string full_path = path + "/" + hname;
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << "<HName>" << hname << "</HName>" << std::endl;
      }
    }
  }
  *out << "</HPathAndHNameList>" << std::endl;
}
//
// -- Select Histograms from Global folder
//
void SiStripInformationExtractor::getGlobalHistos(DQMStore* dqm_store, const std::multimap<std::string, std::string>& req_map, xgi::Output * out) {
 
  std::vector<std::string> hlist;  
  getItemList(req_map,"histo", hlist);

  std::string path = getItemValue(req_map, "GlobalFolder");    

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  std::string opt =" ";

  std::vector<MonitorElement *> all_mes = dqm_store->getContents(path);

  setXMLHeader(out);
  *out << "<HPathAndHNameList>" << std::endl;
  *out << "<HPath>" << path << "</HPath>" << std::endl;

  for (std::vector<std::string>::const_iterator ih = hlist.begin();
       ih != hlist.end(); ih++) {      
    for (std::vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {
      MonitorElement * me = (*it);
      if (!me) continue;
      std::string hname = me->getName();
      std::string name = hname.substr(0, hname.find("__det__"));
      if (name == (*ih)) {
	std::string full_path = path + "/" + hname;
	histoPlotter_->setNewPlot(full_path, opt, width, height);
	*out << "<HName>" << name << "</HName>" << std::endl;
      }
    }
  }
  *out << "</HPathAndHNameList>" << std::endl;
}
//
// -- Get All histograms from a Path
//
void SiStripInformationExtractor::getHistosFromPath(DQMStore * dqm_store, const std::multimap<std::string, std::string>& req_map, xgi::Output * out){

  std::string path = getItemValue(req_map,"Path");

  if (path.size() == 0) return;

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  std::string opt =" ";

  setXMLHeader(out);
  *out << "<HPathAndHNameList>" << std::endl;
  *out << "<HPath>" << path << "</HPath>" << std::endl;

  std::vector<MonitorElement*> all_mes = dqm_store->getContents(path);
  for(std::vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
    MonitorElement* me = (*it);
    if (!me) continue;
    std::string name = me->getName();
    std::string full_path = path + "/" + name;
    histoPlotter_->setNewPlot(full_path, opt, width, height);
    *out << "<HName>" << name << "</HName>" << std::endl;
  }
  *out << "</HPathAndHNameList>" << std::endl;
}
//
// plot Histograms from Layout
//
void SiStripInformationExtractor::plotHistosFromLayout(DQMStore * dqm_store){
  if (layoutMap.size() == 0) return;

  ofstream image_file;
  
  for (std::map<std::string, std::vector< std::string > >::iterator it = layoutMap.begin() ; it != layoutMap.end(); it++) {
    unsigned int ival = 0;
    std::string image_list = "images/" + it->first +".lis";
    image_file.open(image_list.c_str(), std::ios::out);
    if (!image_file) return;

    image_file << "[";
    for (std::vector<std::string>::iterator im = it->second.begin(); 
	 im != it->second.end(); im++) {  
      std::string path_name = (*im);
      if (path_name.size() == 0) continue;
      MonitorElement* me = dqm_store->get(path_name);
      ival++;
      std::ostringstream  fname, ftitle;
      if (!me) {
        fname << "images/EmptyPlot.png";
        ftitle << "EmptyPlot";                 
      } else {
	fname << "images/" << it->first << "_" <<ival << ".png";
        ftitle << me->getName();
        histoPlotter_->createStaticPlot(me, fname.str());
      }
      image_file << "["<< "\"" << fname.str() << "\",\"" << path_name << "\"]";
      if (ival != it->second.size()) image_file << ","<< std::endl;
    }
    image_file << "]" << std::endl;
    image_file.close();
  }
}
//
// -- Plot Tracker Map MEs
//
void SiStripInformationExtractor::getTrackerMapHistos(DQMStore* dqm_store, const std::multimap<std::string, std::string>& req_map, xgi::Output * out) {

  uint32_t detId = atoi(getItemValue(req_map,"ModId").c_str());
 
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  std::string opt =" ";
  
  SiStripFolderOrganizer folder_organizer;
  std::string path;
  folder_organizer.getFolderName(detId,path);   

  std::vector<MonitorElement*> all_mes = dqm_store->getContents(path);
  setXMLHeader(out);
  *out << "<HPathAndHNameList>" << std::endl;
  *out << "<HPath>" << path << "</HPath>" << std::endl;
  for (std::vector<MonitorElement *>::const_iterator it = all_mes.begin();
       it!= all_mes.end(); it++) {
    MonitorElement * me = (*it);
    if (!me) continue;
    std::string hname = me->getName(); 
    std::string full_path;
    full_path = path + "/" + hname;
    histoPlotter_->setNewPlot(full_path, opt, width, height);
    *out << "<HName>" << hname << "</HName>" << std::endl;
  }
  *out << "</HPathAndHNameList>" << std::endl;   

}
//
// -- Select Histograms for a given module
//
void SiStripInformationExtractor::getCondDBHistos(DQMStore * dqm_store, bool& plot_flag, const std::multimap<std::string, std::string>& req_map,xgi::Output * out){

  plot_flag = true;
  std::string sname = getItemValue(req_map,"StructureName");
  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  std::string path = "";
  uint32_t detId;
  if (hasItem(req_map,std::string("ModId"))) {
    detId = atoi(getItemValue(req_map,"ModId").c_str());
    if (detId != 0) {
      SiStripFolderOrganizer folder_organizer;
      folder_organizer.getFolderName(detId,path);
    }
  } else {
    if (sname.size() > 0) path = "SiStrip/" + sname;
  }
  if (path.size() == 0) path = "dummy_path";

  setXMLHeader(out);
  *out << "<HPathAndHNameList>" << std::endl;
  *out << "<HPath>" << path << "</HPath>" << std::endl;
  
  if (path == "dummy_path") {
    *out << "<HName>" << "Dummy" << "</HName>" << std::endl;
    plot_flag = false;
  } else {
    std::string opt = getItemValue(req_map,"option");
    std::vector<std::string> htypes;
    SiStripUtility::split(opt, htypes, ",");
    // Check if CondDB histograms already exists
    std::vector<MonitorElement*> all_mes = dqm_store->getContents(path);
    for (std::vector<std::string>::const_iterator ih = htypes.begin();
	 ih!= htypes.end(); ih++) {
      std::string type = (*ih);
      if (type.size() == 0) continue;
      for (std::vector<MonitorElement *>::const_iterator it = all_mes.begin();
	   it!= all_mes.end(); it++) {
	MonitorElement * me = (*it);
	if (!me) continue;
	std::string hname = me->getName();
	if (hname.find(type) != std::string::npos) {
	  plot_flag = false;
	  break;
	} 
      }  
      if (plot_flag == false) break;
    } 
    histoPlotter_->setNewCondDBPlot(path, opt, width, height);
    for (std::vector<std::string>::const_iterator ih = htypes.begin();
	 ih != htypes.end(); ih++) {
      if ((*ih).size() > 0) {
	*out << "<HName>" << (*ih)  << "</HName>" << std::endl;
      }
    }
  }
  *out << "</HPathAndHNameList>" << std::endl;
}
//
// -- Get a tagged image 
//
void SiStripInformationExtractor::getImage(const std::multimap<std::string, std::string>& req_map, xgi::Output * out){
  
  std::string path = getItemValue(req_map,"Path");
  std::string image;
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
void SiStripInformationExtractor::readLayoutNames(DQMStore* dqm_store, xgi::Output * out){
  setXMLHeader(out);
  *out << "<LayoutAndTKMapList>" << std::endl;
  if (layoutMap.size() > 0) {
    *out << "<LayoutList>" << std::endl;
    for (std::map<std::string, std::vector< std::string > >::iterator it = layoutMap.begin();
	it != layoutMap.end(); it++) {
     *out << "<LName>" << it->first << "</LName>" << std::endl;  
   }
   *out << "</LayoutList>" << std::endl;
  }
  dqm_store->cd();
  *out << "<TKMapOptionList>" << std::endl;
  *out << "<TkMapOption>" << "QTestAlarm" << "</TkMapOption>" << std::endl;

  for (std::vector<std::string>::const_iterator isubdet = subdetVec.begin(); isubdet != subdetVec.end(); isubdet++) {
    std::string dname = (*isubdet);
    if (!dqm_store->dirExists(dname)) continue;
    dqm_store->cd(dname);
    std::vector<std::string> subDirVec = dqm_store->getSubdirs();
    if (subDirVec.size() == 0) continue;
    for (std::vector<std::string>::const_iterator ilayer = subDirVec.begin(); ilayer != subDirVec.end(); ilayer++) {
      std::string lname = (*ilayer);
      if (lname.find("BadModuleList") != std::string::npos) continue; 
      dqm_store->cd(lname);
      break;
    }
    std::vector<std::string> meVec = dqm_store->getMEs();
    for (std::vector<std::string>::const_iterator it = meVec.begin();
	 it != meVec.end(); it++) {
      std::string hname = (*it); 
      if (hname.find("TkHMap_") != std::string::npos) {
	std::string name = hname.substr(hname.find("TkHMap_")+7);
	name = name.substr(0,name.find_first_of("_")); 
	*out << "<TkMapOption>" << name << "</TkMapOption>" << std::endl;
      }
    }
    break;
  }
  *out << "</TKMapOptionList>" << std::endl;      
  *out << "</LayoutAndTKMapList>" << std::endl;
   dqm_store->cd();
}
//
// read the Module And HistoList
//
void SiStripInformationExtractor::readModuleAndHistoList(DQMStore* dqm_store, std::string& sname, const edm::ESHandle<SiStripDetCabling>& detcabling,xgi::Output * out) {

  std::string dname = "SiStrip/" + sname;
  if (!dqm_store->dirExists(dname)) return;

  dqm_store->cd(dname);
  std::vector<std::string> mids; 
  SiStripUtility::getModuleFolderList(dqm_store, mids);
  dqm_store->cd();

  setXMLHeader(out);
  *out << "<ModuleAndHistoList>" << std::endl;
  
  // Fill Module List
  *out << "<ModuleList>" << std::endl;
  uint32_t aDetId  = 0;
  for (std::vector<std::string>::const_iterator it=mids.begin(); it != mids.end(); it++){    
    std::string moduleId = (*it);
    moduleId = moduleId.substr(moduleId.find("module_")+7); 
    *out << "<ModuleNum>" << moduleId << "</ModuleNum>" << std::endl;     
    if (aDetId == 0) aDetId = atoi(moduleId.c_str());
  }
  
  *out << "</ModuleList>" << std::endl;
  // Fill Histo list
  *out << "<HistoList>" << std::endl;
  
  std::vector<MonitorElement*> detector_mes = dqm_store->getContents(mids[0]);
  for (std::vector<MonitorElement *>::const_iterator it = detector_mes.begin();
	 it!= detector_mes.end(); it++) {
    MonitorElement * me = (*it);     
    if (!me) continue;
    std::string hname_full = me->getName();
    std::string hname = hname_full.substr(0, hname_full.find("__det__"));
    *out << "<Histo>" << hname << "</Histo>" << std::endl;     
  }
  *out << "</HistoList>" << std::endl;
  *out << "</ModuleAndHistoList>" << std::endl;
}
//
// Global Histogram List
//
void SiStripInformationExtractor::readGlobalHistoList(DQMStore* dqm_store, std::string& str_name,xgi::Output * out) {
   std::vector<std::string> hnames;
   std::string dname = str_name;
  
   setXMLHeader(out);
   *out << "<GlobalHistoList>" << std::endl;
   if (dqm_store->dirExists(dname)) {
     std::vector<MonitorElement*> meVec = dqm_store->getContents(dname);
     for (std::vector<MonitorElement *>::const_iterator it = meVec.begin();
	  it != meVec.end(); it++) {
       MonitorElement* me = (*it);
       if (!me) continue;
       *out << "<GHisto>" << (*it)->getName() << "</GHisto>" << std::endl;           
     }
   } else {   
     *out << "<GHisto>" << " Desired directory : " << "</GHisto>" << std::endl;
     *out << "<GHisto>" <<       dname             << "</GHisto>" << std::endl;
     *out << "<GHisto>" << " does not exist!!!!  " << "</GHisto>" << std::endl;      
   }
   *out << "</GlobalHistoList>" << std::endl;
}
//
// read the Structure And SummaryHistogram List
//
void SiStripInformationExtractor::readSummaryHistoTree(DQMStore* dqm_store, std::string& str_name, xgi::Output * out) {
  std::ostringstream sumtree;
  std::string dname = "SiStrip/" + str_name;
  if (dqm_store->dirExists(dname)) {    
    dqm_store->cd(dname);
    sumtree << "<ul id=\"summary_histo_tree\" class=\"filetree\">" << std::endl;
    printSummaryHistoList(dqm_store,sumtree);
    sumtree <<"</ul>" << std::endl;   
  } else {
    sumtree << " Desired Directory :  " << std::endl;
    sumtree <<       dname              << std::endl;
    sumtree <<  " does not exist !!!! " << std::endl;
  }
  setPlainHeader(out);
  *out << sumtree.str();
  dqm_store->cd();
}
//
// read the Structure And Alarm Tree
//
void SiStripInformationExtractor::readAlarmTree(DQMStore* dqm_store, 
                  std::string& str_name, xgi::Output * out){
  std::ostringstream alarmtree;
  std::string dname = "SiStrip/" + str_name;
  if (dqm_store->dirExists(dname)) {    
    dqm_store->cd(dname);
    alarmtree << "<ul id=\"alarm_tree\" class=\"filetree\">" << std::endl;
    printAlarmList(dqm_store,alarmtree);
    alarmtree <<"</ul>" << std::endl; 
  } else {
    alarmtree << "Desired Directory :   " << std::endl;
    alarmtree <<       dname              << std::endl;
    alarmtree <<  " does not exist !!!! " << std::endl;
  }
  setPlainHeader(out);
  *out << alarmtree.str();
  dqm_store->cd();
}
//
// Get elements from multi map
//
void SiStripInformationExtractor::getItemList(const std::multimap<std::string, std::string>& req_map, std::string item_name,std::vector<std::string>& items) {
  items.clear();
  for (std::multimap<std::string, std::string>::const_iterator it = req_map.begin();
       it != req_map.end(); it++) {
    
    if (it->first == item_name) {
      items.push_back(it->second);
    }
  }
}
//
//  check a specific item in the map
//
bool SiStripInformationExtractor::hasItem(const std::multimap<std::string,std::string>& req_map,
					  std::string item_name){
  std::multimap<std::string,std::string>::const_iterator pos = req_map.find(item_name);
  if (pos != req_map.end()) return true;
  return false;  
}
//
// check the value of an item in the map
//  
std::string SiStripInformationExtractor::getItemValue(const std::multimap<std::string,std::string>& req_map,
						 std::string item_name){
  std::multimap<std::string,std::string>::const_iterator pos = req_map.find(item_name);
  std::string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
  return value;
}
//
// -- Get color  name from status
//
void SiStripInformationExtractor::selectColor(std::string& col, int status){
  if (status == dqm::qstatus::STATUS_OK)    col = "#00ff00";
  else if (status == dqm::qstatus::WARNING) col = "#ffff00";
  else if (status == dqm::qstatus::ERROR)   col = "#ff0000";
  else if (status == dqm::qstatus::OTHER)   col = "#ffa500";
  else  col = "#0000ff";
}
//
// -- Get Image name from ME
//
void SiStripInformationExtractor::selectColor(std::string& col, std::vector<QReport*>& reports){
  int istat = 999;
  int status = 0;
  for (std::vector<QReport*>::const_iterator it = reports.begin(); it != reports.end();
       it++) {
    status = (*it)->getStatus();
    if (status > istat) istat = status;
  }
  selectColor(col, status);
}
//
// -- Get Image name from status
//
void SiStripInformationExtractor::selectImage(std::string& name, int status){
  if (status == dqm::qstatus::STATUS_OK) name="images/LI_green.gif";
  else if (status == dqm::qstatus::WARNING) name="images/LI_yellow.gif";
  else if (status == dqm::qstatus::ERROR) name="images/LI_red.gif";
  else if (status == dqm::qstatus::OTHER) name="images/LI_orange.gif";
  else  name="images/LI_blue.gif";
}
//
// -- Get Image name from ME
//
void SiStripInformationExtractor::selectImage(std::string& name, std::vector<QReport*>& reports){
  int istat = 999;
  int status = 0;
  for (std::vector<QReport*>::const_iterator it = reports.begin(); it != reports.end();
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

  std::string path = getItemValue(req_map,"Path");

  int width  = atoi(getItemValue(req_map, "width").c_str());
  int height = atoi(getItemValue(req_map, "height").c_str());

  std::string opt =" ";

  std::ostringstream test_status;
  
  setXMLHeader(out);
  *out << "<StatusAndPath>" << std::endl;
  *out << "<PathList>" << std::endl;
  if (path.size() == 0) {
    *out << "<HPath>" << "NONE" << "</HPath>" << std::endl;     
    test_status << " ME Does not exist ! " << std::endl;
  } else {
    std::vector<MonitorElement*> all_mes = dqm_store->getContents(path);
    *out << "<HPath>" << path << "</HPath>" << std::endl;     
    for(std::vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
      MonitorElement* me = (*it);
      if (!me) continue;
      std::string name = me->getName();  

      std::vector<QReport*> q_reports = me->getQReports();
      if (q_reports.size() == 0 && name.find("StripQualityFromCondDB") == std::string::npos) continue;
      std::string full_path = path + "/" + name;
      histoPlotter_->setNewPlot(full_path, opt, width, height);

      if (q_reports.size() != 0) {
        test_status << " QTest Status for " << name << " : " << std::endl;
        test_status << " ======================================================== " << std::endl; 
        for (std::vector<QReport*>::const_iterator it = q_reports.begin(); it != q_reports.end();
	     it++) {
	  int status = (*it)->getStatus();
	  if (status == dqm::qstatus::WARNING) test_status << " Warning ";
	  else if (status == dqm::qstatus::ERROR) test_status << " Error  ";
	  else if (status == dqm::qstatus::STATUS_OK) test_status << " Ok  ";
	  else if (status == dqm::qstatus::OTHER) test_status << " Other(" << status << ") ";
	  std::string mess_str = (*it)->getMessage();
	  test_status <<  " &lt;br/&gt;";
	  mess_str = mess_str.substr(mess_str.find(" Test")+5);
	  test_status <<  " QTest Name  : " << mess_str.substr(0, mess_str.find(")")+1) << std::endl;
	  test_status << " &lt;br/&gt;";
	  test_status <<  " QTest Detail  : " << mess_str.substr(mess_str.find(")")+2) << std::endl;
	} 
	test_status << " ======================================================== " << std::endl;
      }
      *out << "<HName>" << name << "</HName>" << std::endl;         
    }    
  }
  *out << "</PathList>" << std::endl;
  *out << "<StatusList>" << std::endl;
  *out << "<Status>" << test_status.str() << "</Status>" << std::endl;      
  *out << "</StatusList>" << std::endl;
  *out << "</StatusAndPath>" << std::endl;
}
//
// -- Read the text Summary of QTest result
//
void SiStripInformationExtractor::readQTestSummary(DQMStore* dqm_store, std::string type, xgi::Output * out) {

  int nDetsWithError = 0;
  int nDetsTotal = 0;
  std::ostringstream qtest_summary, lite_summary;
  
  dqm_store->cd();
  SiStripFolderOrganizer folder_organizer;
  for (std::vector<std::string>::const_iterator isubdet = subdetVec.begin(); isubdet != subdetVec.end(); isubdet++) {
    std::string dname = (*isubdet);
    std::string bad_module_folder = dname + "/" + "BadModuleList";
    if (!dqm_store->dirExists(dname)) continue;

    dqm_store->cd(dname);
    std::vector<std::string> mids;
    SiStripUtility::getModuleFolderList(dqm_store, mids);
  
    int ndet    = mids.size();
    int errdet = 0;
    dqm_store->cd();
    dqm_store->cd(dname);

    qtest_summary << dname.substr(dname.find("View/")+5) << " : <br/>";
    qtest_summary << "=============================="<< "<br/>";
    if (dqm_store->dirExists(bad_module_folder)) {

      std::vector<MonitorElement *> meVec = dqm_store->getContents(bad_module_folder);
      for (std::vector<MonitorElement *>::const_iterator it = meVec.begin();
           it != meVec.end(); it++) {
        errdet++;
        int flag = (*it)->getIntValue();
	std::string message;
	SiStripUtility::getBadModuleStatus(flag, message);
        qtest_summary << " Module Id " << (*it)->getName() << " has "<< message << "<br/>";
      }
    }
    qtest_summary << "--------------------------------------------------------------------"<< "<br/>";
    qtest_summary << " Detectors :  Total "<< ndet
		  << " with Error " << errdet << "<br/>";
    qtest_summary << "--------------------------------------------------------------------"<< "<br/>";
    nDetsWithError += errdet;
    nDetsTotal     += ndet;
  }
  qtest_summary << "--------------------------------------------------------------------"<< "<br/>";
  qtest_summary << "--------------------------------------------------------------------"<< "<br/>";
  qtest_summary << " Total Detectors " << nDetsTotal;
  qtest_summary << " # of Detectors with Error " << nDetsWithError << "<br/>";
  qtest_summary << "--------------------------------------------------------------------"<< "<br/>";
  qtest_summary << "--------------------------------------------------------------------"<< "<br/>";

  lite_summary << " Total Detectors " << nDetsTotal << "<br/>";
  lite_summary << " # of Detectors with Error " << nDetsWithError << "<br/>";

  setHTMLHeader(out);
  if (type == "Lite") *out << lite_summary.str();
  else {
    *out << qtest_summary.str();
  }
  dqm_store->cd();
}
//
// -- Create Images 
//
void SiStripInformationExtractor::createImages(DQMStore* dqm_store){
  if (histoPlotter_->plotsToMake())       histoPlotter_->createPlots(dqm_store);
  if (histoPlotter_->condDBPlotsToMake()) histoPlotter_->createCondDBPlots(dqm_store);
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
void SiStripInformationExtractor::readNonGeomHistoTree(DQMStore* dqm_store, std::string& fld_name, xgi::Output * out) {
  std::ostringstream sumtree;
  std::string dname = "SiStrip/" + fld_name;
  if (dqm_store->dirExists(dname)) {    
    dqm_store->cd(dname);
    sumtree << "<ul id=\"non_geo_tree\" class=\"filetree\">" << std::endl;
    printNonGeomHistoList(dqm_store,sumtree);
    sumtree <<"</ul>" << std::endl;   
  } else {
    sumtree << " Desired Directory :  " << std::endl;
    sumtree <<       dname              << std::endl;
    sumtree <<  " does not exist !!!! " << std::endl;
  }
  setPlainHeader(out);
  *out << sumtree.str();
  dqm_store->cd();
}
//
// --  Fill Readout/Control Histo List
// 
void SiStripInformationExtractor::printNonGeomHistoList(DQMStore * dqm_store, std::ostringstream& str_val){
  static std::string indent_str = "";

  std::string currDir = dqm_store->pwd();
  std::string dname = currDir.substr(currDir.find_last_of("/")+1);
  str_val << "<li><span class=\"folder\">" << dname << "</span>" << std::endl;
  std::vector<MonitorElement *> meVec = dqm_store->getContents(currDir);
  std::vector<std::string> subDirVec = dqm_store->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << "</li> "<< std::endl;    
    return;
  }
  str_val << "<ul>" << std::endl;      
  for (std::vector<MonitorElement *>::const_iterator it = meVec.begin();
         it != meVec.end(); it++) {
    MonitorElement* me = (*it);
    if (!me) continue;
    std::string name = (*it)->getName();
    str_val << "<li> <span class=\"file\"><a href=\"javascript:RequestHistos.DrawSummaryHistogram('" 
            << currDir
	    << "')\">" << name << "</a></span></li>" << std::endl;
  }
  for (std::vector<std::string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    dqm_store->cd(*ic);
    printNonGeomHistoList(dqm_store, str_val);
    dqm_store->goUp();
  }
  str_val << "</ul> "<< std::endl;  
  str_val << "</li> "<< std::endl;  
}


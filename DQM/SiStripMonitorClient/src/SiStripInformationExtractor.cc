#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"

#include "TText.h"
#include "TROOT.h"
#include "TPad.h"
#include "TSystem.h"
#include "TString.h"
#include "TImage.h"
#include "TPaveText.h"
#include "TImageDump.h"

#include <iostream>
using namespace std;

//
// -- Constructor
// 
SiStripInformationExtractor::SiStripInformationExtractor() {
  edm::LogInfo("SiStripInformationExtractor") << 
    " Creating SiStripInformationExtractor " << "\n" ;
}
//
// --  Destructor
// 
SiStripInformationExtractor::~SiStripInformationExtractor() {
  edm::LogInfo("SiStripInformationExtractor") << 
    " Deleting SiStripInformationExtractor " << "\n" ;
  //  if (theCanvas) delete theCanvas;
}
//
// --  Fill Histo and Module List
// 
void SiStripInformationExtractor::fillModuleAndHistoList(MonitorUserInterface * mui, vector<string>& modules, vector<string>& histos) {
  string currDir = mui->pwd();
  if (currDir.find("module_") != string::npos)  {
    string mId = currDir.substr(currDir.find("module")+7, 9);
    modules.push_back(mId);
    if (histos.size() == 0) {
      vector<string> contents = mui->getMEs();    
      for (vector<string>::const_iterator it = contents.begin();
	   it != contents.end(); it++) {
	string hname = (*it).substr(0, (*it).find("__det__"));
        histos.push_back(hname);
      }    
    }
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      fillModuleAndHistoList(mui, modules, histos);
      mui->goUp();
    }
  }
}
//
// --  Fill Summary Histo List
// 
void SiStripInformationExtractor::fillSummaryHistoList(MonitorUserInterface * mui, string& str_name, vector<string>& histos){
  string currDir = mui->pwd();
  vector<string> contentVec;
  mui->getContents(contentVec);
  
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    if ((*it).find(str_name) == string::npos) continue;
    if ((*it).find("module_") != string::npos) continue;
    vector<string> contents;
    int nval = SiStripUtility::getMEList((*it), contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator ic = contents.begin();
	 ic != contents.end(); ic++) {
      if ((*ic).find("Summary") != string::npos) {
        string me_name =  (*ic).substr((*ic).find(str_name)+str_name.size()+1);
        histos.push_back(me_name);
      }
    }
  }
}
//
// --  Get Selected Monitor Elements
// 
void SiStripInformationExtractor::selectSingleModuleHistos(MonitorUserInterface * mui, string mid, vector<string>& names, vector<MonitorElement*>& mes) {
  string currDir = mui->pwd();
  if (currDir.find("module_") != string::npos &&
      currDir.find(mid) != string::npos )  {
    vector<string> contents = mui->getMEs();    
    for (vector<string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      for (vector<string>::const_iterator ih = names.begin();
	   ih != names.end(); ih++) {
	string temp_s = (*it).substr(0, (*it).find("__"));
	//	if ((*it).find((*ih)) != string::npos) {
	if (temp_s == (*ih)) {
	  string full_path = currDir + "/" + (*it);
	  MonitorElement * me = mui->get(full_path.c_str());
	  if (me) mes.push_back(me);
	}  
      }
    }
    if (mes.size() >0) return;
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      selectSingleModuleHistos(mui, mid, names, mes);
      mui->goUp();
    }
  }
}
//
// --  Get Selected Summary Monitor Elements
// 
void SiStripInformationExtractor::selectSummaryHistos(MonitorUserInterface * mui,string str_name, vector<string>& names, vector<MonitorElement*>& mes) {
  
  vector<string> contentVec;
  mui->getContents(contentVec);
  
  for (vector<string>::iterator it = contentVec.begin();
       it != contentVec.end(); it++) {
    if ((*it).find(str_name) == string::npos) continue;
    if ((*it).find("module_") != string::npos) continue;
    vector<string> contents;
    string dir_path;
    int nval = SiStripUtility::getMEList((*it), dir_path, contents);
    if (nval == 0) continue;
    for (vector<string>::const_iterator ic = contents.begin();
	 ic != contents.end(); ic++) {
      for (vector<string>::const_iterator im = names.begin();
	   im != names.end(); im++) {
        string hname = (*im).substr((*im).find_last_of("/")+1);
	if ((*ic).find(hname) != string::npos) {
	  string full_path = dir_path + "/" + (*ic);
	  MonitorElement * me = mui->get(full_path.c_str());
	  if (me) mes.push_back(me);
	}
      }
    }
    if ( mes.size() == names.size()) return;
  }
}
//
// --  Plot Selected Monitor Elements
// 
void SiStripInformationExtractor::plotSingleModuleHistos(MonitorUserInterface* mui, multimap<string, string>& req_map) {

  vector<string> item_list;  

  string mod_id = getItemValue(req_map,"ModId");
  if (mod_id.size() < 9) return;
  item_list.clear();     
  getItemList(req_map,"histo", item_list);

  vector<MonitorElement*> me_list;

  mui->cd();
  selectSingleModuleHistos(mui, mod_id, item_list, me_list);
  mui->cd();

  plotHistos(req_map,me_list);
}
//
// -- plot Summary Histos
//
void SiStripInformationExtractor::plotSummaryHistos(MonitorUserInterface * mui,
		       std::multimap<std::string, std::string>& req_map){
  vector<string> item_list;  

  string str_name = getItemValue(req_map,"StructureName");
  if (str_name.size() == 0) return;
  item_list.clear();     
  getItemList(req_map,"histo", item_list);

  vector<MonitorElement*> me_list;

  mui->cd();
  selectSummaryHistos(mui, str_name, item_list, me_list);
  mui->cd();

  plotHistos(req_map,me_list);
}
//
//  plot Histograms in a Canvas
//
void SiStripInformationExtractor::plotHistos(multimap<string,string>& req_map, 
			   vector<MonitorElement*> me_list){
  if (me_list.size() == 0) return;
  TCanvas canvas("TestCanvas", "Test Canvas",600, 600);
  canvas.Clear();
  int ncol, nrow;
 
  float xlow = -1.0;
  float xhigh = -1.0;
  
  if (me_list.size() == 1) {
    if (hasItem(req_map,"xmin")) xlow = atof(getItemValue(req_map,"xmin").c_str());
    if (hasItem(req_map,"xmax")) xhigh = atof(getItemValue(req_map,"xmax").c_str()); 
    ncol = 1;
    nrow = 1;
  } else {
    ncol = atoi(getItemValue(req_map, "cols").c_str());
    nrow = atoi(getItemValue(req_map, "rows").c_str());
  }
  canvas.Divide(ncol, nrow);
  int i=0;
  for (vector<MonitorElement*>::const_iterator it = me_list.begin();
       it != me_list.end(); it++) {
    i++;
    MonitorElementT<TNamed>* ob = 
      dynamic_cast<MonitorElementT<TNamed>*>((*it));
    if (ob) {
      canvas.cd(i);
      //      TAxis* xa = ob->operator->()->GetXaxis();
      //      xa->SetRangeUser(xlow, xhigh);
      ob->operator->()->Draw();
      if (hasItem(req_map,"logy")) {
	  gPad->SetLogy(1);
      }
    }
  }
  canvas.Update();
  fillImageBuffer(canvas);
  canvas.Clear();
}
//
// read the Module And HistoList
//
void SiStripInformationExtractor::readModuleAndHistoList(MonitorUserInterface* mui, xgi::Output * out, bool coll_flag) {
   std::vector<std::string> hnames;
   std::vector<std::string> mod_names;
   if (coll_flag)  mui->cd("Collector/Collated");
   fillModuleAndHistoList(mui, mod_names, hnames);
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
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
   if (coll_flag)  mui->cd();
}
//
// read the Structure And SummaryHistogram List
//
void SiStripInformationExtractor::readSummaryHistoList(MonitorUserInterface* mui, std::string& str_name, xgi::Output * out, bool coll_flag) {

   if (coll_flag)  mui->cd("Collector/Collated");
   std::vector<std::string> hnames;
   fillSummaryHistoList(mui, str_name, hnames);
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
   *out << "<?xml version=\"1.0\" ?>" << std::endl;
   *out << "<SummaryHistoList>" << endl;

   for (std::vector<std::string>::iterator ih = hnames.begin();
        ih != hnames.end(); ih++) {
     *out << "<SummaryHisto>" << *ih << "</SummaryHisto>" << endl;     
   }
   *out << "</SummaryHistoList>" << endl;
   if (coll_flag)  mui->cd();
}
//
// Get elements from multi map
//
void SiStripInformationExtractor::getItemList(multimap<string, string>& req_map, 
                      string item_name,vector<string>& items) {
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
string SiStripInformationExtractor::getItemValue(multimap<string,string>& req_map,
						 std::string item_name){
  multimap<string,string>::iterator pos = req_map.find(item_name);
  string value = " ";
  if (pos != req_map.end()) {
    value = pos->second;
  }
  return value;
}
//
// write the canvas in a string
//
void SiStripInformationExtractor::fillImageBuffer(TCanvas& c1) {

  c1.SetFixedAspectRatio(kTRUE);
  c1.SetCanvasSize(520, 440);
  // Now extract the image
  // 114 - stands for "no write on Close"
  TImageDump imgdump("tmp.png", 114);
  c1.Paint();

 // get an internal image which will be automatically deleted
 // in the imgdump destructor
  TImage *image = imgdump.GetImage();

  char *buf;
  int sz;
  image->GetImageBuffer(&buf, &sz);         /* raw buffer */

  pictureBuffer_.str("");
  for (int i = 0; i < sz; i++)
    pictureBuffer_ << buf[i];
  
  delete [] buf;
}
//
// get the plot
//
const ostringstream&  SiStripInformationExtractor::getImage() const {
  return pictureBuffer_;
}

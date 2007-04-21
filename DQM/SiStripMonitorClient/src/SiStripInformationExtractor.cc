#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "TText.h"
#include "TROOT.h"
#include "TPad.h"
#include "TSystem.h"
#include "TString.h"
#include "TImage.h"
#include "TPaveText.h"
#include "TImageDump.h"
#include "TAxis.h"
#include "TStyle.h"

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
  readConfiguration();
  canvas_ = new TCanvas("TestCanvas", "Test Canvas");
}
//
// --  Destructor
// 
SiStripInformationExtractor::~SiStripInformationExtractor() {
  edm::LogInfo("SiStripInformationExtractor") << 
    " Deleting SiStripInformationExtractor " << "\n" ;
  if (layoutParser_) delete layoutParser_;
  if (canvas_) delete canvas_;

}
//
// -- Read Configurationn File
//
void SiStripInformationExtractor::readConfiguration() {
  string localPath = string("DQM/SiStripMonitorClient/test/sistrip_plot_layout.xml");
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
// --  Fill Histo and Module List
// 
void SiStripInformationExtractor::fillModuleAndHistoList(MonitorUserInterface * mui, vector<string>& modules, vector<string>& histos) {
  string currDir = mui->pwd();
  if (currDir.find("module_") != string::npos)  {
    string mId = currDir.substr(currDir.find("module_")+7, 9);
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
void SiStripInformationExtractor::printSummaryHistoList(MonitorUserInterface * mui, ostringstream& str_val){
  static string indent_str = "";

  string currDir = mui->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  if (dname.find("module_") ==0) return;
  str_val << "<li><a href=\"#\" id=\"" 
          << currDir << "\">" << dname << "</a>" << endl;
  vector<string> meVec = mui->getMEs(); 
  vector<string> subDirVec = mui->getSubdirs();
  if ( meVec.size()== 0  && subDirVec.size() == 0 ) {
    str_val << "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;      
  for (vector<string>::const_iterator it = meVec.begin();
       it != meVec.end(); it++) {
    if ((*it).find("Summary") == 0) {
      str_val << "<li class=\"dhtmlgoodies_sheet.gif\"><a href=\"javascript:DrawSingleHisto('"
           << currDir << "/"<< (*it) << "')\">" << (*it) << "</a></li>" << endl;
    }
  }

  string mtag ="Modules: ";  
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    mui->cd(*ic);
    string titl = (*ic);
    if (titl.find("module_") == 0)  {
      titl = titl.substr(titl.find("module_")+7);
      mtag += titl + " ";
    }
    printSummaryHistoList(mui, str_val);
    mui->goUp();
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
void SiStripInformationExtractor::printAlarmList(MonitorUserInterface * mui, ostringstream& str_val){
  static string indent_str = "";

  string currDir = mui->pwd();
  string dname = currDir.substr(currDir.find_last_of("/")+1);
  string image_name;
  selectImage(image_name,mui->getStatus(currDir));
  str_val << "<li><a href=\"#\" id=\"" 
          << currDir << "\">" << dname << "</a> <img src=\"" 
          << image_name << "\">" << endl;
  vector<string> subDirVec = mui->getSubdirs();
  vector<string> meVec = mui->getMEs(); 
  if (subDirVec.size() == 0 && meVec.size() == 0) {
    str_val << "</li> "<< endl;    
    return;
  }
  str_val << "<ul>" << endl;
  if (dname.find("module_") != string::npos) {
    if (meVec.size() > 0) {
      for (vector<string>::const_iterator it = meVec.begin();
	   it != meVec.end(); it++) {
        string full_path = currDir + "/" + (*it);
	MonitorElement * me = mui->get(full_path);
	if (!me) continue;
        dqm::qtests::QR_map my_map = me->getQReports();
        if (my_map.size() > 0) {
	  string image_name1;
	  selectImage(image_name1,my_map);
	  str_val << "<li class=\"dhtmlgoodies_sheet.gif\"><a href=\"javascript:ReadStatus('"
		<< full_path<< "')\">" << (*it) << "</a><img src=\""
		<< image_name1 << "\""<< "</li>" << endl;
        }
      }
    }
  }
  for (vector<string>::const_iterator ic = subDirVec.begin();
       ic != subDirVec.end(); ic++) {
    mui->cd(*ic);
    printAlarmList(mui, str_val);
    mui->goUp();
  }
  str_val << "</ul> "<< endl;  
  str_val << "</li> "<< endl;  
}
//
// --  Fill Histo and Module List
// 
void SiStripInformationExtractor::fillGlobalHistoList(MonitorUserInterface * mui, vector<string>& histos) {
  string currDir = mui->pwd();
  if (currDir.find("GlobalParameters") != string::npos)  {
    vector<string> contents = mui->getMEs();    
    for (vector<string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      histos.push_back((*it));
    }
    return;
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      fillGlobalHistoList(mui,histos);
      mui->goUp();
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
// --  Get Selected Monitor Elements
// 
void SiStripInformationExtractor::selectGlobalHistos(MonitorUserInterface * mui, vector<string>& names, vector<MonitorElement*>& mes) {
  string currDir = mui->pwd();
  if (currDir.find("GlobalParameters") != string::npos)  {
    vector<string> contents = mui->getMEs();    
    for (vector<string>::const_iterator it = contents.begin();
	 it != contents.end(); it++) {
      for (vector<string>::const_iterator ih = names.begin();
	   ih != names.end(); ih++) {
	if ((*it) == (*ih)) {
	  string full_path = currDir + "/" + (*it);
	  MonitorElement * me = mui->get(full_path.c_str());
	  if (me) mes.push_back(me);
        }
      }
    }
    return;
  } else {  
    vector<string> subdirs = mui->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      mui->cd(*it);
      selectGlobalHistos(mui,names, mes);
      mui->goUp();
    }
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

  plotHistos(req_map,me_list,false);
  fillImageBuffer();
  canvas_->Clear();

}
//
// --  Plot Selected Monitor Elements
// 
void SiStripInformationExtractor::plotGlobalHistos(MonitorUserInterface* mui, multimap<string, string>& req_map) {
 
  vector<string> item_list;  

  item_list.clear();     
  getItemList(req_map,"histo", item_list);
  vector<MonitorElement*> me_list;

  mui->cd();
  selectGlobalHistos(mui, item_list, me_list);
  mui->cd();

  plotHistos(req_map,me_list,false);
  fillImageBuffer();
  canvas_->Clear();

}
//
// -- plot a Histogram
//
void SiStripInformationExtractor::plotHistosFromPath(MonitorUserInterface * mui, std::multimap<std::string, std::string>& req_map){
  vector<string> item_list;  
  getItemList(req_map,"Path", item_list);
  
  if (item_list.size() == 0) return;

  vector<MonitorElement*> me_list;
  string htype  = getItemValue(req_map,"histotype");
  if (htype.size() == 0) htype="individual";

  for (vector<string>::iterator it = item_list.begin(); it != item_list.end(); it++) {  

    string path_name = (*it);
    if (path_name.size() == 0) continue;
    
    MonitorElement* me = mui->get(path_name);

    if (me) me_list.push_back(me);
  }
  if (me_list.size() == 0) return; 
  if (htype == "summary") plotHistos(req_map, me_list, true);
  else plotHistos(req_map, me_list, false); 

  fillImageBuffer();
  canvas_->Clear();

}
//
// plot Histograms from Layout
//
void SiStripInformationExtractor::plotHistosFromLayout(MonitorUserInterface * mui){
  if (layoutMap.size() == 0) return;
  multimap<string, string> opt_map;
  opt_map.insert(pair<string,string>("width","600"));
  opt_map.insert(pair<string,string>("height","600"));

  for (map<std::string, std::vector< std::string > >::iterator it = layoutMap.begin() ; it != layoutMap.end(); it++) {
    string fname  = it->first + ".png";
    vector < MonitorElement* > me_list;
    for (vector<string>::iterator im = it->second.begin(); 
	 im != it->second.end(); im++) {  
      string path_name = (*im);
      if (path_name.size() == 0) continue;
      MonitorElement* me = mui->get(path_name);
      if (me) me_list.push_back(me);
    }
    if (me_list.size() > 0) { 
      plotHistos(opt_map, me_list, false); 
      canvas_->Print(fname.c_str(),"png");
      canvas_->Clear();
      me_list.clear();
    }
  }
}

//
//  plot Histograms in a Canvas
//
void SiStripInformationExtractor::plotHistos(multimap<string,string>& req_map, 
  			   vector<MonitorElement*> me_list, bool sflag){
  int nhist = me_list.size();
  if (nhist == 0) return;
  int width = 900;
  int height = 900;
  canvas_->Clear();
  int ncol = 1, nrow = 1;
 
  float xlow = -1.0;
  float xhigh = -1.0;
  
  if (nhist == 1) {
    if (hasItem(req_map,"xmin")) xlow = atof(getItemValue(req_map,"xmin").c_str());
    if (hasItem(req_map,"xmax")) xhigh = atof(getItemValue(req_map,"xmax").c_str()); 
    ncol = 1;
    if (sflag) nrow = 2;
    else nrow = 1;
  } else {
    if (hasItem(req_map,"cols")) ncol = atoi(getItemValue(req_map, "cols").c_str());
    if (hasItem(req_map,"rows")) nrow = atoi(getItemValue(req_map, "rows").c_str());
    if (ncol*nrow < nhist) {
      if (nhist == 2) {
	ncol = 1;
	nrow = 2;
      } else if (nhist == 3) {
	ncol = 1;
	nrow = 3;
      } else if (nhist == 4) {
	ncol = 2;
	nrow = 2;
      } else if (nhist == 5 ||nhist == 6 ) {
	ncol = 2;
	nrow = 3;
      } else if (nhist == 7 ||nhist == 8 ) {
	ncol = 2;
	nrow = 4;
      } else if (nhist > 8 && nhist <= 12) {
        ncol = 3;
	nrow = nhist/ncol+1;
      } else if (nhist > 10 && nhist <= 20) {
        ncol = 3;
	nrow = nhist/ncol+1;
      } else if (nhist > 20 && nhist <= 40) {
         ncol = 4;
	 nrow = nhist/ncol+1;
      } 		

    }
  }
  if (hasItem(req_map,"width")) 
              width = atoi(getItemValue(req_map, "width").c_str());    
  if (hasItem(req_map,"height"))
              height = atoi(getItemValue(req_map, "height").c_str());

  string dopt;
  if (hasItem(req_map,"drawopt")) dopt = getItemValue(req_map, "drawopt");

  canvas_->SetWindowSize(width,height);
  canvas_->Divide(ncol, nrow);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("emruo");
  gStyle->SetStatFontSize(0.05);
  int idir=0;
  for (vector<MonitorElement*>::const_iterator it = me_list.begin();
       it != me_list.end(); it++) {
    idir++;
    int istat =  SiStripUtility::getStatus((*it));
    string tag;
    int icol;
    SiStripUtility::getStatusColor(istat, icol, tag);

    TProfile* prof = ExtractTObject<TProfile>().extract((*it));
    TH1F* hist1 = ExtractTObject<TH1F>().extract((*it));
    TH2F* hist2 = ExtractTObject<TH2F>().extract((*it));
  
    if (prof|| hist1 || hist2) {
      canvas_->cd(idir);
      TText tTitle;
      tTitle.SetTextFont(64);
      tTitle.SetTextSizePixels(20);
      if (hist2) {
        if (xlow != -1.0 && xhigh != -1.0) {
          TAxis* xa = hist2->GetXaxis();
          xa->SetRangeUser(xlow, xhigh);
        }
        hist2->SetFillColor(1);
        if (dopt.find("projection") != string::npos) {
          TH1F thproj(hist2->GetName(),hist2->GetTitle(),hist2->GetNbinsY(), 
	      hist2->GetYaxis()->GetXmin(),hist2->GetYaxis()->GetXmax());
	  for (int j = 1; j < hist2->GetNbinsY()+1; j++) {
	    for (int i = 1; i < hist2->GetNbinsX()+1; i++) {
	      thproj.SetBinContent(j, hist2->GetBinContent(i,j));
            }
	  }
          thproj.DrawCopy();
	} else hist2->Draw(dopt.c_str());
        tTitle.DrawTextNDC(0.1, 0.92, hist2->GetName());
      } else if (prof) {
        if (xlow != -1 &&  xhigh != -1.0) {
          TAxis* xa = prof->GetXaxis();
          xa->SetRangeUser(xlow, xhigh);
        }
        if (dopt.find("projection") != string::npos) {
          TH1F thproj(prof->GetName(),prof->GetTitle(),100, 
		      0.0,prof->GetMaximum()*1.2);
          for (int i = 1; i < prof->GetNbinsX()+1; i++) {
	    thproj.Fill(prof->GetBinContent(i));
	  }
	  thproj.DrawCopy();
        } else prof->Draw(dopt.c_str());
        tTitle.DrawTextNDC(0.1, 0.92, prof->GetName());
      } else {
        if (xlow != -1 &&  xhigh != -1.0) {
          TAxis* xa = hist1->GetXaxis();
          xa->SetRangeUser(xlow, xhigh);
        }
        if (dopt.find("projection") != string::npos || sflag) {
          TH1F thproj(hist1->GetName(),hist1->GetTitle(),100, 
		      0.0,hist1->GetMaximum()*1.2);
          for (int i = 1; i < hist1->GetNbinsX()+1; i++) {
	    thproj.Fill(hist1->GetBinContent(i));
	  }
	  if (sflag) {
	    hist1->Draw();
	    canvas_->cd(idir+1); 	
            gPad->SetLogy(1);
            thproj.DrawCopy();
	  }
          else thproj.DrawCopy();
        } else hist1->Draw();
        tTitle.DrawTextNDC(0.1, 0.92, hist1->GetName());
      }
      if (icol != 1) {
	TText tt;
	tt.SetTextSize(0.12);
	tt.SetTextColor(icol);
	tt.DrawTextNDC(0.5, 0.5, tag.c_str());
      }
      if (hasItem(req_map,"logy")) {
	gPad->SetLogy(1);
      }
    }
  }
  canvas_->Update();
  canvas_->Modified();
}
//
// -- Read Layout Group names
//
void SiStripInformationExtractor::readLayoutNames(xgi::Output * out){
  if (layoutMap.size() > 0) {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
    *out << "<?xml version=\"1.0\" ?>" << std::endl;
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
// read the Module And HistoList
//
void SiStripInformationExtractor::readGlobalHistoList(MonitorUserInterface* mui, xgi::Output * out, bool coll_flag) {
   if (coll_flag)  mui->cd("Collector/Collated");
   std::vector<std::string> hnames;

   fillGlobalHistoList(mui, hnames);
   
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
   *out << "<?xml version=\"1.0\" ?>" << std::endl;
   *out << "<GlobalHistoList>" << endl;
   for (vector<string>::const_iterator it = hnames.begin();
	it != hnames.end(); it++) {
     *out << "<GHisto>" << *it << "</GHisto>" << endl;      
   }
   *out << "</GlobalHistoList>" << endl;
   if (coll_flag)  mui->cd();
}
//
// read the Structure And SummaryHistogram List
//
void SiStripInformationExtractor::readSummaryHistoTree(MonitorUserInterface* mui, string& str_name, xgi::Output * out, bool coll_flag) {
 
  ostringstream sumtree;
  if (goToDir(mui, str_name, coll_flag)) {
    sumtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printSummaryHistoList(mui,sumtree);
    sumtree <<"</ul>" << endl;   
  } else {
    sumtree << "Desired Directory does not exist";
  }
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  *out << sumtree.str();
   mui->cd();
}
//
// read the Structure And Alarm Tree
//
void SiStripInformationExtractor::readAlarmTree(MonitorUserInterface* mui, 
                  string& str_name, xgi::Output * out, bool coll_flag){
  ostringstream alarmtree;
  if (goToDir(mui, str_name, coll_flag)) {
    alarmtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printAlarmList(mui,alarmtree);
    alarmtree <<"</ul>" << endl; 
  } else {
    alarmtree << "Desired Directory does not exist";
  }
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  *out << alarmtree.str();
   mui->cd();
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
void SiStripInformationExtractor::fillImageBuffer() {

  canvas_->SetFixedAspectRatio(kTRUE);
  //  canvas_->SetCanvasSize(520, 440);
  // Now extract the image
  // 114 - stands for "no write on Close"
  TImageDump imgdump("tmp.png", 114);
  canvas_->Paint();

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
//
// go to a specific directory after scanning
//
bool SiStripInformationExtractor::goToDir(MonitorUserInterface* mui, string& sname, bool flg){ 
  mui->cd();
  mui->cd("Collector");
  cout << mui->pwd() << endl;
  vector<string> subdirs;
  subdirs = mui->getSubdirs();
  if (subdirs.size() == 0) return false;
  
  if (flg) mui->cd("Collated");
  else mui->cd(subdirs[0]);
  cout << mui->pwd() << endl;
  subdirs.clear();
  subdirs = mui->getSubdirs();
  if (subdirs.size() == 0) return false;
  mui->cd(sname);
  string dirName = mui->pwd();
  if (dirName.find(sname) != string::npos) return true;
  else return false;  
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
void SiStripInformationExtractor::selectImage(string& name, dqm::qtests::QR_map& test_map){
  int istat = 999;
  int status = 0;
  for (dqm::qtests::QR_map::const_iterator it = test_map.begin(); it != test_map.end();
       it++) {
    status = it->second->getStatus();
    if (status > istat) istat = status;
  }
  selectImage(name, status);
}
//
// -- Get Warning/Error Messages
//
void SiStripInformationExtractor::readStatusMessage(MonitorUserInterface* mui, string& path,xgi::Output * out) {
  MonitorElement* me = mui->get(path);
  string hpath;
  ostringstream test_status;
  if (!me) {
    test_status << " ME Does not exist ! ";
    hpath = "NOME";
  } else {
    hpath = path.substr(0, path.find("."));
    dqm::qtests::QR_map test_map = me->getQReports();
    for (dqm::qtests::QR_map::const_iterator it = test_map.begin(); it != test_map.end();
	 it++) {
      int status = it->second->getStatus();
      test_status << " QTest Status : ";
      if (status == dqm::qstatus::WARNING) test_status << " Warning ";
      else if (status == dqm::qstatus::ERROR) test_status << " Error  ";
      else if (status == dqm::qstatus::STATUS_OK) test_status << " Ok  ";
      else if (status == dqm::qstatus::OTHER) test_status << " Other(" << status << ") ";
      string mess_str = it->second->getMessage();
      test_status <<  "&lt;br/&gt;";
      mess_str = mess_str.substr(mess_str.find(" Test")+5);
      test_status <<  " QTest Name  : " << mess_str.substr(0, mess_str.find(")")+1) << endl;
      test_status << "&lt;br/&gt;";
      test_status <<  " QTest Detail  : " << mess_str.substr(mess_str.find(")")+2) << endl;      
    }      
  }
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
   *out << "<?xml version=\"1.0\" ?>" << std::endl;
   *out << "<StatusAndPath>" << endl;
   *out << "<StatusList>" << endl;
   *out << "<Status>" << test_status.str() << "</Status>" << endl;      
   *out << "</StatusList>" << endl;
   *out << "<PathList>" << endl;
   *out << "<HPath>" << hpath << "</HPath>" << endl;   
   *out << "</PathList>" << endl;
   *out << "</StatusAndPath>" << endl;
}


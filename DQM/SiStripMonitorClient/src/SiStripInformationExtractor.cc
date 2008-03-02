#include "DQM/SiStripMonitorClient/interface/SiStripInformationExtractor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/QTest.h"
#include "DQMServices/Core/interface/QReport.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorClient/interface/SiStripLayoutParser.h"
#include "DQM/SiStripMonitorClient/interface/SiStripConfigParser.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"


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
#include "TPaveLabel.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include <iostream>
using namespace std;

//
// -- Constructor
// 
SiStripInformationExtractor::SiStripInformationExtractor() {
  edm::LogInfo("SiStripInformationExtractor") << 
    " Creating SiStripInformationExtractor " << "\n" ;
  layoutParser_ = 0;
  readReference_ = false;
  layoutMap.clear();
  canvasWidth  = 600;
  canvasHeight = 400;
  canvas_ = new TCanvas("TestCanvas", "Test Canvas", canvasWidth, canvasHeight); 
  readConfiguration();
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

  // No need of dummies as Emppty plots are alrady there
  //      createDummiesFromLayout();
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
    if (name.find("Summary") == 0) {
      str_val << "<li class=\"dhtmlgoodies_sheet.gif\">"
              << " <a href=\"javascript:RequestHistos.DrawSingleHisto('"
              << currDir 
              << "')\">" << name 
              << "</a></li>" << endl;
    }
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
// --  Get Selected Monitor Elements
// 
void SiStripInformationExtractor::selectSingleModuleHistos(DQMStore * dqm_store, string mid, vector<string>& names, vector<MonitorElement*>& mes) {
  mes.clear();
  unsigned int tag = atoi(mid.c_str());
  vector<MonitorElement*> all_mes = dqm_store->get(tag);
  if (all_mes.size() == 0) return; 
  for (vector<MonitorElement *>::const_iterator it = all_mes.begin();
       it!= all_mes.end(); it++) {
    if (!(*it)) continue;
    for (vector<string>::const_iterator ih = names.begin();
	 ih != names.end(); ih++) {
      
      string me_name = (*it)->getName();
      if (me_name.find(*ih) != string::npos) {
	mes.push_back((*it));
      }
    }
  }
}
//
// --  Get Selected Monitor Elements
// 
void SiStripInformationExtractor::selectGlobalHistos(DQMStore * dqm_store, string dname, vector<string>& names, vector<MonitorElement*>& mes) {
  if (dqm_store->dirExists(dname)) {
    vector<MonitorElement*> meVec = dqm_store->getContents(dname);
    for (vector<MonitorElement *>::const_iterator it = meVec.begin();
	 it != meVec.end(); it++) {
      MonitorElement* me = (*it);
      if (!me) continue;
      for (vector<string>::const_iterator ih = names.begin();
	   ih != names.end(); ih++) {
	if (me->getName() == (*ih)) mes.push_back(me);
      }
    }       
  }   
}
//
// --  Plot Selected Monitor Elements
// 
void SiStripInformationExtractor::plotSingleModuleHistos(DQMStore* dqm_store, multimap<string, string>& req_map, xgi::Output * out) {

  setCanvasDimension(req_map);

  vector<string> item_list;  

  string mod_id = getItemValue(req_map,"ModId");
  if (mod_id.size() < 9) {
    setCanvasMessage("Wrong Module Id!!");
    fillImageBuffer();
    canvas_->Clear();
    return;
  }
  item_list.clear();     
  getItemList(req_map,"histo", item_list);
  vector<MonitorElement*> me_list;

  selectSingleModuleHistos(dqm_store, mod_id, item_list, me_list);
  if (namedPictureBuffer_.size() > 100) namedPictureBuffer_.clear();
  if (me_list.size() == 0) {
    setCanvasMessage("Wrong Module Id!!");  
  } else {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
    out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    string path = me_list[0]->getPathname();
    *out << path << " ";
    for (vector<MonitorElement*>::iterator it = me_list.begin(); it != me_list.end(); it++) {
      MonitorElement* me = (*it);
      if (!me) continue;
      canvas_->Clear();
      plotHisto(req_map, me, false);
      string name = me->getName();
      fillNamedImageBuffer(me->getFullname());
      *out << name << " ";
    }
  }
}
//
// --  Plot Selected Monitor Elements
// 
void SiStripInformationExtractor::plotGlobalHistos(DQMStore* dqm_store, multimap<string, string>& req_map, xgi::Output * out) {
 
  setCanvasDimension(req_map);

  vector<string> item_list;  

  item_list.clear();     
  getItemList(req_map,"histo", item_list);

  string dname = "SiStrip/" +  getItemValue(req_map, "GlobalFolder");    

  vector<MonitorElement *> me_list;
  selectGlobalHistos(dqm_store, dname, item_list, me_list);
  if (me_list.size() == 0) {
    setCanvasMessage("Requested Histograms do not exist!!");
  } else {
    out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
    out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
    out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
    out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
    string path = me_list[0]->getPathname();
    *out << path << " ";
    for (vector<MonitorElement*>::iterator it = me_list.begin(); it != me_list.end(); it++) {
      MonitorElement* me = (*it);
      if (!me) continue;
      canvas_->Clear();
      plotHisto(req_map, me, false);
      string name = me->getName();
      fillNamedImageBuffer(me->getFullname());
      *out << name << " ";
    }
  }
}
//
// -- plot a Histogram
//
void SiStripInformationExtractor::plotHistosFromPath(DQMStore * dqm_store, std::multimap<std::string, std::string>& req_map, xgi::Output * out){

  setCanvasDimension(req_map);

  string me_folder = getItemValue(req_map,"Path");

  if (me_folder.size() == 0) return;

  string htype  = getItemValue(req_map,"histotype");

  if (namedPictureBuffer_.size() > 100) namedPictureBuffer_.clear();

  if (htype.size() == 0) htype="individual";
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/html");
  out->getHTTPResponseHeader().addHeader("Pragma", "no-cache");   
  out->getHTTPResponseHeader().addHeader("Cache-Control", "no-store, no-cache, must-revalidate,max-age=0");
  out->getHTTPResponseHeader().addHeader("Expires","Mon, 26 Jul 1997 05:00:00 GMT");
  vector<MonitorElement*> all_mes = dqm_store->getContents(me_folder);
  *out << me_folder << " " ;
  for(vector<MonitorElement*>::iterator it=all_mes.begin(); it!=all_mes.end(); it++){
    MonitorElement* me = (*it);
    if (!me) continue;
    canvas_->Clear();
    if (htype == "summary") plotHisto(req_map, me, true);
    else  plotHisto(req_map, me, false);
    fillNamedImageBuffer(me->getFullname());
    string name = me->getName();
    *out << name << " ";
  }
}
//
// plot Histograms from Layout
//
void SiStripInformationExtractor::plotHistosFromLayout(DQMStore * dqm_store){
  if (layoutMap.size() == 0) return;
  if (!readReference_) {
    string localPath = string("DQM/SiStripMonitorClient/test/Reference.root");
    dqm_store->open(edm::FileInPath(localPath).fullPath(), false, "", "SiStrip/Reference");
    readReference_ = true;
  }

  canvas_->SetFixedAspectRatio(kTRUE);
  canvas_->SetCanvasSize(canvasWidth, canvasHeight);
  canvas_->Update();
  canvas_->Clear();

  gStyle->SetOptStat("emruo");
  gStyle->SetStatFontSize(0.05);
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
    gStyle->SetOptTitle(0);
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
 	TH1* hist1 = me->getTH1();
	if (hist1) {
	  
	  TText tTitle;
	  tTitle.SetTextFont(64);
	  tTitle.SetTextSizePixels(20);
	  
          setDrawingOption(hist1);
	  hist1->DrawCopy();
	  
          string hname = hist1->GetTitle();
	  string ref_path = "SiStrip/Reference/" + it->first + "/" +hname;
	  
          tTitle.DrawTextNDC(0.1, 0.92, hname.c_str());
	  MonitorElement* me_ref = dqm_store->get(ref_path);
	  if (me_ref) {
	    TH1* hist1_ref = me_ref->getTH1();
	    if (hist1_ref) {
//	      cout << ref_path << " " << hist1_ref->GetEntries() << " " << 
endl;
	      hist1_ref->SetLineColor(3);
              hist1_ref->SetMarkerColor(3);
              if (hname.find("Summary") != string::npos) hist1_ref->DrawCopy("same");
              else hist1_ref->DrawNormalized("same", hist1->GetEntries());
            }
	  }
	}
	string command = "rm -f " + fname.str();
	gSystem->Exec(command.c_str());
	canvas_->Print(fname.str().c_str(),"png");
	canvas_->Clear();	
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
void SiStripInformationExtractor::plotTrackerMapHistos(DQMStore* dqm_store, std::multimap<std::string, std::string>& req_map, xgi::Output * out) {

  vector<string> me_names;
  string tkmap_name;
  SiStripConfigParser config_parser;
  string localPath = string("DQM/SiStripMonitorClient/data/sistrip_monitorelement_config.xml");
  config_parser.getDocument(edm::FileInPath(localPath).fullPath());
  if (!config_parser.getMENamesForTrackerMap(tkmap_name, me_names));
  if (me_names.size() == 0) return;
  for (vector<string>::iterator it = me_names.begin();
       it != me_names.end(); it++) {
    req_map.insert(pair<string,string>("histo",(*it)));  
  }   
  plotSingleModuleHistos(dqm_store, req_map, out);
}

//
// -- Plot Dummy Histograms from Layout
//
void SiStripInformationExtractor::createDummiesFromLayout(){
  canvas_->SetFixedAspectRatio(kTRUE);
  canvas_->SetCanvasSize(canvasWidth, canvasHeight);
  canvas_->Update();
  canvas_->Clear();
  if (layoutMap.size() == 0) return;
  for (map<std::string, std::vector< std::string > >::iterator it = layoutMap.begin() ; it != layoutMap.end(); it++) {
    string fname  = it->first + ".png";
    setCanvasMessage("Plot not ready yet!!");
    canvas_->Print(fname.c_str(),"png");
    canvas_->Clear();
  }
}
//
// -- Get a tagged image 
//
const ostringstream& SiStripInformationExtractor::getIMGCImage(DQMStore * dqm_store, std::multimap<std::string, std::string>& req_map){
  
  string path = getItemValue(req_map,"Path");
  string meName;
  if (hasNamedImage(path)) return getNamedImage(path);
  else {
    canvas_->Clear();
    setCanvasMessage("Plot does not exist !");
    fillImageBuffer();
    canvas_->Clear();
    return getImage();
  }

}
//
//  plot Histograms in a Canvas
//
void SiStripInformationExtractor::plotHistos(multimap<string,string>& req_map, 
  			   vector<MonitorElement*> me_list, bool sflag){
  int nhist = me_list.size();
  if (nhist == 0) return;
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
    if (ncol*nrow < nhist) defineZone(nhist, ncol, nrow);
  }

  string dopt;
  if (hasItem(req_map,"drawopt")) dopt = getItemValue(req_map, "drawopt");

  canvas_->Divide(ncol, nrow);
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("emruo");
  gStyle->SetStatFontSize(0.05);

  int idir=0;
  for (vector<MonitorElement*>::const_iterator it = me_list.begin();
       it != me_list.end(); it++) {
    idir++;
    int istat =  SiStripUtility::getMEStatus((*it));
    string tag;
    int icol;
    SiStripUtility::getMEStatusColor(istat, icol, tag);
    MonitorElement* me = (*it);
    if (!me) continue;
    if (me->getTH1()) {
      TProfile* prof = 0;
      TH1F* hist1 = 0;
      TH2F* hist2 = 0;
      
      canvas_->cd(idir);
      TText tTitle;
      tTitle.SetTextFont(64);
      tTitle.SetTextSizePixels(20);
      if (me->kind() == MonitorElement::DQM_KIND_TH2F) {
	hist2 = me->getTH2F();
	setDrawingOption(hist2, xlow, xhigh);
	hist2->SetFillColor(1);
	if (dopt.find("projection") != string::npos) {
	  string ptit = hist2->GetTitle();
	  ptit += " (y-projection)";
	  TH1F thproj(hist2->GetName(),ptit.c_str(),hist2->GetNbinsY(), 
		      hist2->GetYaxis()->GetXmin(),hist2->GetYaxis()->GetXmax());
	  thproj.GetXaxis()->SetTitle(ptit.c_str());
	  for (int j = 1; j < hist2->GetNbinsY()+1; j++) {
	    float tot_count = 0.0;
	    for (int i = 1; i < hist2->GetNbinsX()+1; i++) {
	      tot_count += hist2->GetBinContent(i,j);
	    }
	    thproj.SetBinContent(j, tot_count);
	  }
          thproj.DrawCopy();
	} else hist2->Draw(dopt.c_str());
        tTitle.DrawTextNDC(0.1, 0.92, hist2->GetName());
      } else if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) {
	prof = me->getTProfile();
	setDrawingOption(prof, xlow, xhigh);
	if (dopt.find("projection") != string::npos) {
	  
	  string ptit = hist1->GetTitle();
	  ptit += " (y-projection)";
	  TH1F thproj(prof->GetName(),ptit.c_str(),100, 
		      0.0,prof->GetMaximum()*1.2);
	  thproj.GetXaxis()->SetTitle(ptit.c_str());
	  for (int i = 1; i < prof->GetNbinsX()+1; i++) {
	    thproj.Fill(prof->GetBinContent(i));
	  }
	  thproj.DrawCopy();
	} else prof->Draw(dopt.c_str());
	tTitle.DrawTextNDC(0.1, 0.92, prof->GetName());
      } else if  (me->kind() == MonitorElement::DQM_KIND_TH1F) {
	hist1 = me->getTH1F();
	setDrawingOption(hist1, xlow, xhigh);
	if (dopt.find("projection") != string::npos || sflag) {
	  string ptit = hist1->GetTitle();
	  ptit += " (y-projection)";
	  TH1F thproj(hist1->GetName(),ptit.c_str(),100, 
		      0.0,hist1->GetMaximum()*1.2);
	  thproj.GetXaxis()->SetTitle(ptit.c_str());
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
    } else setCanvasMessage("Plot does not exist (yet)!!"); 
  }
  canvas_->Update();
  canvas_->Modified();
}
//
//  plot Histograms in a Canvas
//
void SiStripInformationExtractor::plotHisto(multimap<string,string>& req_map, 
  			   MonitorElement* me, bool sflag){
  if (sflag) canvas_->Divide(1,2);

  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("emruo");
  gStyle->SetStatFontSize(0.05);

  int istat =  SiStripUtility::getMEStatus(me);
  string tag;
  int icol;
  SiStripUtility::getMEStatusColor(istat, icol, tag);

  float xlow = -1.0;
  float xhigh = -1.0;
  
  if (hasItem(req_map,"xmin")) xlow = atof(getItemValue(req_map,"xmin").c_str());
  if (hasItem(req_map,"xmax")) xhigh = atof(getItemValue(req_map,"xmax").c_str()); 

  string dopt;
  if (hasItem(req_map,"drawopt")) dopt = getItemValue(req_map, "drawopt");

  if (me->getTH1()) {
    TProfile* prof = 0;
    TH1F* hist1 = 0; 
    TH2F* hist2 = 0;
    
    TText tTitle;
    tTitle.SetTextFont(64);
    tTitle.SetTextSizePixels(20);
    if (me->kind() == MonitorElement::DQM_KIND_TH2F) {
      hist2 = me->getTH2F();
      setDrawingOption(hist2, xlow, xhigh);
      hist2->SetFillColor(1);
      if (dopt.find("projection") != string::npos) {
	string ptit = hist2->GetTitle();
	ptit += " (y-projection)";
	TH1F thproj(hist2->GetName(),ptit.c_str(),hist2->GetNbinsY(), 
		    hist2->GetYaxis()->GetXmin(),hist2->GetYaxis()->GetXmax());
	thproj.GetXaxis()->SetTitle(ptit.c_str());
	for (int j = 1; j < hist2->GetNbinsY()+1; j++) {
	  float tot_count = 0.0;
	  for (int i = 1; i < hist2->GetNbinsX()+1; i++) {
	    tot_count += hist2->GetBinContent(i,j);
	  }
	  thproj.SetBinContent(j, tot_count);
	}
	thproj.DrawCopy();
      } else hist2->Draw(dopt.c_str());
      tTitle.DrawTextNDC(0.1, 0.92, hist2->GetName());
    } else if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) {
      prof = me->getTProfile();
      setDrawingOption(prof, xlow, xhigh);
      if (dopt.find("projection") != string::npos) {
	
	string ptit = hist1->GetTitle();
	ptit += " (y-projection)";
	TH1F thproj(prof->GetName(),ptit.c_str(),100, 
		    0.0,prof->GetMaximum()*1.2);
	thproj.GetXaxis()->SetTitle(ptit.c_str());
	for (int i = 1; i < prof->GetNbinsX()+1; i++) {
	  thproj.Fill(prof->GetBinContent(i));
	}
	thproj.DrawCopy();
      } else prof->Draw(dopt.c_str());
      tTitle.DrawTextNDC(0.1, 0.92, prof->GetName());
    } else if  (me->kind() == MonitorElement::DQM_KIND_TH1F) {
      hist1 = me->getTH1F();
      if (dopt.find("projection") != string::npos || sflag) {
	string ptit = hist1->GetTitle();
	ptit += " (y-projection)";
	TH1F thproj(hist1->GetName(),ptit.c_str(),100, 
		    0.0,hist1->GetMaximum()*1.2);
	thproj.GetXaxis()->SetTitle(ptit.c_str());
	for (int i = 1; i < hist1->GetNbinsX()+1; i++) {
	  thproj.Fill(hist1->GetBinContent(i));
	}
	if (sflag) {
	  canvas_->cd(1); 	
	  hist1->Draw();
	  canvas_->cd(2); 	
	  gPad->SetLogy(1);
	  thproj.DrawCopy();
	} else thproj.DrawCopy();
      } else hist1->Draw();
      tTitle.DrawTextNDC(0.1, 0.92, hist1->GetName());
    }
    if (icol != 1) {
      TText tt;
      tt.SetTextSize(0.12);
      tt.SetTextColor(icol);
      tt.DrawTextNDC(0.5, 0.5, tag.c_str());
    }
  } else setCanvasMessage("Plot does not exist (yet)!!"); 

  canvas_->Update();
  canvas_->Modified();
}
//
// -- Read Layout Group names
//
void SiStripInformationExtractor::readLayoutNames(multimap<string, string>& req_map, xgi::Output * out){
  
  setCanvasDimension(req_map);

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
void SiStripInformationExtractor::readModuleAndHistoList(DQMStore* dqm_store, const edm::ESHandle<SiStripDetCabling>& detcabling, xgi::Output * out) {

  std::vector<uint32_t> SelectedDetIds;
  detcabling->addActiveDetectorsRawIds(SelectedDetIds);

   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<ModuleAndHistoList>" << endl;


  // Fill Module List
  *out << "<ModuleList>" << endl;
  uint32_t aDetId  = 0;
  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){    
    uint32_t detId = *idetid;
    if (detId == 0 || detId == 0xFFFFFFFF){
      edm::LogError("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::createMEs: " 
					       << "Wrong DetId !!!!!! " <<  detId << " Neglecting !!!!!! ";
      continue;
    }
    if (aDetId == 0) aDetId = detId;
    ostringstream detIdStr;
    detIdStr << detId;
    *out << "<ModuleNum>" << detIdStr.str() << "</ModuleNum>" << endl;     
  }
  *out << "</ModuleList>" << endl;
  // Fill Histo list
  *out << "<HistoList>" << endl;
  vector<MonitorElement*> detector_mes = dqm_store->get(aDetId);
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
*/
}
//
// Global Histogram List
//
void SiStripInformationExtractor::readGlobalHistoList(DQMStore* dqm_store, std::string& str_name,xgi::Output * out) {
   std::vector<std::string> hnames;
   string dname = "SiStrip/" + str_name;
   out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
   *out << "<?xml version=\"1.0\" ?>" << std::endl;
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
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
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
    //  if (goToDir(dqm_store, str_name)) {
    alarmtree << "<ul id=\"dhtmlgoodies_tree\" class=\"dhtmlgoodies_tree\">" << endl;
    printAlarmList(dqm_store,alarmtree);
    alarmtree <<"</ul>" << endl; 
  } else {
    alarmtree << "Desired Directory :   " << endl;
    alarmtree <<       dname              << endl;
    alarmtree <<  " does not exist !!!! " << endl;
  }
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  *out << alarmtree.str();
   dqm_store->cd();
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

//  canvas_->SetFixedAspectRatio(kTRUE);
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
bool SiStripInformationExtractor::goToDir(DQMStore* dqm_store, string& sname){ 
  string currDir = dqm_store->pwd();
  if (currDir.find("SiStrip") != string::npos)  {
    dqm_store->cd(sname);
    string dname = dqm_store->pwd();
    if (dname.find(sname) != string::npos) return true;
    else return false;
  } else {
    vector<string> subdirs = dqm_store->getSubdirs();
    for (vector<string>::const_iterator it = subdirs.begin();
	 it != subdirs.end(); it++) {
      dqm_store->cd(*it);
      if (goToDir(dqm_store, sname)) return true;
      else dqm_store->goUp();
    }
    return false;
  }  
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

  string me_folder = getItemValue(req_map,"Path");
  ostringstream test_status;
  
  out->getHTTPResponseHeader().addHeader("Content-Type", "text/xml");
  *out << "<?xml version=\"1.0\" ?>" << std::endl;
  *out << "<StatusAndPath>" << endl;
  *out << "<PathList>" << endl;
  if (me_folder.size() == 0) {
    *out << "<HPath>" << "NONE" << "</HPath>" << endl;     
    test_status << " ME Does not exist ! " << endl;
  } else {
    vector<MonitorElement*> all_mes = dqm_store->getContents(me_folder);
    *out << "<HPath>" << me_folder << "</HPath>" << endl;     
    for(vector<MonitorElement*>::iterator ic=all_mes.begin(); ic!=all_mes.end(); ic++){
      MonitorElement* me = (*ic);
      if (!me) continue;
      string name = me->getName();  

      vector<QReport*> q_reports = me->getQReports();
      if (q_reports.size() == 0 && name.find("StripQualityFromCondDB") == string::npos) continue;
      canvas_->Clear();
      plotHisto(req_map, me, false);
      fillNamedImageBuffer(me->getFullname());

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
  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){    
    uint32_t detId = *idetid;
    if (detId == 0 || detId == 0xFFFFFFFF){
      edm::LogError("SiStripMonitorPedestals") <<"SiStripMonitorPedestals::createMEs: " 
					       << "Wrong DetId !!!!!! " <<  detId << " Neglecting !!!!!! ";
      continue;
    }
    nDetsTotal++;
    vector<MonitorElement*> detector_mes = dqm_store->get(detId);
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


  out->getHTTPResponseHeader().addHeader("Content-Type", "text/plain");
  if (type == "Lite") *out << lite_summary.str();
  else {
   if (nDetsWithWarning == 0 && nDetsWithError ==0)  *out << lite_summary.str();
   else  *out << qtest_summary.str();
  }

  dqm_store->cd();
}
//
// -- Define Zone from # of histograms
//
void SiStripInformationExtractor::defineZone(int nhist, int& ncol, int & nrow) {

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
//
// -- Set Canvas Message
//
void SiStripInformationExtractor::setCanvasMessage(const string& error_string) {
  TText tLabel;
  tLabel.SetTextSize(0.16);
  tLabel.SetTextColor(4);
  tLabel.DrawTextNDC(0.1, 0.5, error_string.c_str());
}
//
// -- Set Drawing Option
//
void SiStripInformationExtractor::setDrawingOption(TH1* hist, float xlow, float xhigh) {
  if (!hist) return;

  TAxis* xa = hist->GetXaxis();
  TAxis* ya = hist->GetYaxis();

  xa->SetTitleOffset(0.7);
  xa->SetTitleSize(0.05);
  xa->SetLabelSize(0.04);

  ya->SetTitleOffset(0.7);
  ya->SetTitleSize(0.05);
  ya->SetLabelSize(0.04);

  if (xlow != -1 &&  xhigh != -1.0) {
    xa->SetRangeUser(xlow, xhigh);
  }

}
//
// -- Set Canvas Dimension
//
void SiStripInformationExtractor::setCanvasDimension(std::multimap<std::string, std::string>& req_map) {

  if (hasItem(req_map,"width")) 
              canvasWidth = atoi(getItemValue(req_map, "width").c_str());    
  if (hasItem(req_map,"height"))
              canvasHeight = atoi(getItemValue(req_map, "height").c_str());
  canvas_->SetFixedAspectRatio(kTRUE);
  canvas_->SetCanvasSize(canvasWidth, canvasHeight);
  canvas_->Update();
  canvas_->Clear();
}
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiStripInformationExtractor::fillNamedImageBuffer(std::string name) {
  //  canvas_->SetFixedAspectRatio(kTRUE);
  //  canvas_->SetCanvasSize(520, 440);
  // Now extract the image
  // 114 - stands for "no write on Close"
  TImageDump imgdump("tmp.png", 114);
  canvas_->Paint();

 // get an internal image which will be automatically deleted
 // in the imgdump destructor
  TImage *image = imgdump.GetImage();
  if( image == NULL )
  {
   cout << "SiStripInformationExtractor::fillNamedImageBuffer : "
   	<< "WARNING: " 
	<< "No TImage found for "
	<< name
   	<< endl ;
    return ;
  }

  char *buf;
  int sz = 0;
  image->GetImageBuffer(&buf, &sz);         /* raw buffer */

  pictureBuffer_.str("");
  for (int i = 0; i < sz; i++) pictureBuffer_ << buf[i];
  
  //  delete [] buf;
  ::free(buf); // buf is allocated via realloc() by a C language AfterStep library invoked by the
               // default (and so far only) TImage implementation in root, TASImage.
  
  if (hasNamedImage(name)) namedPictureBuffer_.erase(name);
  namedPictureBuffer_[name] = pictureBuffer_.str();
}
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
const ostringstream& SiStripInformationExtractor::getNamedImage(string name) {

  pictureBuffer_.str("");
  map<string, string>::const_iterator cPos = namedPictureBuffer_.find(name);
  if (cPos == namedPictureBuffer_.end()) { 
  } else pictureBuffer_ << cPos->second;
  return pictureBuffer_;
}
//------------------------------------------------------------------------------
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
bool SiStripInformationExtractor::hasNamedImage(string name) {
  map<string, string>::const_iterator cPos = namedPictureBuffer_.find(name);
  if (cPos == namedPictureBuffer_.end()) { 
    return false;
  } else return true;
}

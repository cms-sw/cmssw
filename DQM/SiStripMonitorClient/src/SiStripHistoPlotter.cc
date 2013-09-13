#include "DQM/SiStripMonitorClient/interface/SiStripHistoPlotter.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
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
//
// -- Constructor
// 
SiStripHistoPlotter::SiStripHistoPlotter() {
  edm::LogInfo("SiStripHistoPlotter") << 
    " Creating SiStripHistoPlotter " << "\n" ;
}
//
// --  Destructor
// 
SiStripHistoPlotter::~SiStripHistoPlotter() {
  edm::LogInfo("SiStripHistoPlotter") << 
    " Deleting SiStripHistoPlotter " << "\n" ;
  plotList_.clear();
  condDBPlotList_.clear();

}
//
// -- Set New Plot
//
void SiStripHistoPlotter::setNewPlot(std::string& path, std::string& option, int width, int height) {
  std::string name = "Dummy";
  if (!hasNamedImage(name)) createDummyImage(name);
  PlotParameter local_par;
  local_par.Path    = path;
  local_par.Option  = option;
  local_par.CWidth  = width;
  local_par.CHeight = height;
  plotList_.push_back(local_par);  
}
//
// -- Create Plots 
//
void SiStripHistoPlotter::createPlots(DQMStore* dqm_store) {
  if (plotList_.size() == 0) return;
  std::string name = "Dummy";
  if (!hasNamedImage(name)) createDummyImage(name);
  for (std::vector<PlotParameter>::iterator it = plotList_.begin(); 
       it != plotList_.end(); it++) {
    makePlot(dqm_store, (*it));
  }
  plotList_.clear();
}
//
// -- Draw Histograms 
//
void SiStripHistoPlotter::makePlot(DQMStore* dqm_store, const PlotParameter& par) {
  TCanvas * canvas = new TCanvas("TKCanvas", "TKCanvas", par.CWidth, par.CHeight);

  MonitorElement * me = dqm_store->get(par.Path);
  if (me) { 
	
    int istat =  SiStripUtility::getMEStatus(me);
    
    std::string dopt = par.Option;
    std::string tag;
    int icol;
    SiStripUtility::getMEStatusColor(istat, icol, tag);
    if (me->kind() == MonitorElement::DQM_KIND_TH1F ||
	me->kind() == MonitorElement::DQM_KIND_TH2F ||
	me->kind() == MonitorElement::DQM_KIND_TPROFILE ||
        me->kind() == MonitorElement::DQM_KIND_TPROFILE2D ) {
      TH1* histo = me->getTH1();
      TH1F* tproject = 0;
      if (dopt == "projection") {
	getProjection(me, tproject);
	if (tproject) tproject->Draw();
	else histo->Draw();
      } else {
	dopt = ""; 
	std::string name = histo->GetName();
        if (me->kind() == MonitorElement::DQM_KIND_TPROFILE2D ) {
          dopt = "colz";
          histo->SetStats( kFALSE );
	} else {    
	  if (name.find("Summary_Mean") != std::string::npos) {
	    histo->SetStats( kFALSE );
	  } else {
	    histo->SetFillColor(1);
	  }
        }
	histo->Draw(dopt.c_str());
      }
    }
    TText tTitle;
    tTitle.SetTextFont(64);
    tTitle.SetTextSizePixels(20);
    //    tTitle.DrawTextNDC(0.1, 0.92, histo->GetName());
    
    if (icol != 1) {
      TText tt;
      tt.SetTextSize(0.12);
      tt.SetTextColor(icol);
      tt.DrawTextNDC(0.5, 0.5, tag.c_str());
    }
    fillNamedImageBuffer(canvas, par.Path);
    canvas->Clear();
  }
  delete canvas;
}
//
// -- Get Named Image buffer
//
void SiStripHistoPlotter::getNamedImageBuffer(const std::string& path, std::string& image) {
  std::map<std::string, std::string>::iterator cPos;
  if (path == "dummy_path") {
    std::cout << " Sending Dummy Image for : "
         <<  path << std::endl;
    cPos = namedPictureBuffer_.find("Dummy");
    image = cPos->second;
  } else {
    cPos = namedPictureBuffer_.find(path);
    if (cPos != namedPictureBuffer_.end()) {
      image = cPos->second;
      if (namedPictureBuffer_.size() > 99 ) namedPictureBuffer_.erase(cPos);
    } else {
      std::cout << " Sending Dummy Image for : "
	   <<  path << std::endl;
      cPos = namedPictureBuffer_.find("Dummy");
      image = cPos->second;
    }
  }
}
/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiStripHistoPlotter::fillNamedImageBuffer(TCanvas * c1, const std::string& name) 
{
  //  DQMScope enter;
 // Now extract the image
  // 114 - stands for "no write on Close"
//   std::cout << ACYellow << ACBold
//        << "[SiPixelInformationExtractor::fillNamedImageBuffer()] "
//        << ACPlain
//        << "A canvas: "
//        << c1->GetName() 
//        << std::endl ;
  c1->Update();
  c1->Modified(); 
  TImageDump imgdump("tmp.png", 114);
  c1->Paint();

// get an internal image which will be automatically deleted
// in the imgdump destructor
  TImage *image = imgdump.GetImage();

  if( image == NULL )
  {
   std::cout << "No TImage found for "
	<< name
   	<< std::endl ;
    return ;
  }
  char *buf;
  int sz = 0;
  image->GetImageBuffer(&buf, &sz);

  std::ostringstream local_str;
  for (int i = 0; i < sz; i++) local_str << buf[i];
  
//  delete [] buf;
  ::free(buf); // buf is allocated via realloc() by a C language AfterStep library invoked by the
               // default (and so far only) TImage implementation in root, TASImage.
  
  // clear the first element map if # of entries > 30
  if (hasNamedImage(name)) namedPictureBuffer_.erase(name);
  namedPictureBuffer_[name] = local_str.str();
  //  if (namedPictureBuffer_[name].size() > 0) std::cout << "image created " << name << std::endl;
}
//
// -- Check if the image exists
//
bool SiStripHistoPlotter::hasNamedImage(const std::string& name) {
  std::map<std::string, std::string>::const_iterator cPos = namedPictureBuffer_.find(name);
  if (cPos == namedPictureBuffer_.end()) { 
    return false;
  } else return true;
}
//
// -- Create Dummy Image 
//
void SiStripHistoPlotter::createDummyImage(const std::string & name) {
  std::string image;
  getDummyImage(image);
  namedPictureBuffer_.insert(std::pair<std::string, std::string>(name, image));
}
//
// -- Get Image reading a disk resident image
//
void SiStripHistoPlotter::getDummyImage(std::string & image) {
  std::string          line;
  std::ostringstream   local_str;
  // Read back the file line by line and temporarily store it in a stringstream
  std::string localPath = std::string("DQM/TrackerCommon/test/images/EmptyPlot.png");
  std::ifstream * imagefile = new std::ifstream((edm::FileInPath(localPath).fullPath()).c_str(),std::ios::in);
  if(imagefile->is_open()) {
    while (getline( *imagefile, line )) {
      local_str << line << std::endl ;
    }
  }  
  imagefile->close();
  image = local_str.str();
}
// -- Set Drawing Option
//
void SiStripHistoPlotter::setDrawingOption(TH1* hist) {
  if (!hist) return;

  TAxis* xa = hist->GetXaxis();
  TAxis* ya = hist->GetYaxis();

  xa->SetTitleOffset(0.7);
  xa->SetTitleSize(0.05);
  xa->SetLabelSize(0.04);

  ya->SetTitleOffset(0.7);
  ya->SetTitleSize(0.05);
  ya->SetLabelSize(0.04);

}
// -- Get Projection Histogram 
//
void SiStripHistoPlotter::getProjection(MonitorElement* me, TH1F* tp) {
  
  std::string ptit = me->getTitle();
  ptit += "-Yprojection";

  if (me->kind() == MonitorElement::DQM_KIND_TH2F) {
    TH2F* hist2 = me->getTH2F();
    tp = new TH1F(ptit.c_str(),ptit.c_str(),hist2->GetNbinsY(), 
		  hist2->GetYaxis()->GetXmin(),hist2->GetYaxis()->GetXmax());
    tp->GetXaxis()->SetTitle(ptit.c_str());
    for (int j = 1; j < hist2->GetNbinsY()+1; j++) {
      float tot_count = 0.0;
      for (int i = 1; i < hist2->GetNbinsX()+1; i++) {
	tot_count += hist2->GetBinContent(i,j);
      }
      tp->SetBinContent(j, tot_count);
    }
  } else if (me->kind() == MonitorElement::DQM_KIND_TPROFILE) {
     TProfile* prof = me->getTProfile();
     tp = new TH1F(ptit.c_str(),ptit.c_str(),100, 
		    0.0,prof->GetMaximum()*1.2);
     tp->GetXaxis()->SetTitle(ptit.c_str());
     for (int i = 1; i < prof->GetNbinsX()+1; i++) {
       tp->Fill(prof->GetBinContent(i));
     }
  } else if (me->kind() == MonitorElement::DQM_KIND_TH1F) {
    TH1F* hist1 = me->getTH1F();
    tp = new TH1F(ptit.c_str(),ptit.c_str(),100, 
	      0.0,hist1->GetMaximum()*1.2);
    tp->GetXaxis()->SetTitle(ptit.c_str());
    for (int i = 1; i < hist1->GetNbinsX()+1; i++) {
      tp->Fill(hist1->GetBinContent(i));
    }
  }
}
//
// -- create static plots
// 
void SiStripHistoPlotter::createStaticPlot(MonitorElement* me, const std::string& file_name) {
  TH1* hist1 = me->getTH1();
  TCanvas* canvas  = new TCanvas("TKCanvas", "TKCanvas", 600, 400);
  if (hist1) {
    TText tTitle;
    tTitle.SetTextFont(64);
    tTitle.SetTextSizePixels(20);
    
    setDrawingOption(hist1);
    hist1->Draw();
    std::string name = hist1->GetName();
    if (me->getRefRootObject()) {
      TH1* hist1_ref = me->getRefTH1();
      if (hist1_ref) {
	hist1_ref->SetLineColor(3);
	hist1_ref->SetMarkerColor(3);
	if (name.find("Summary") != std::string::npos) hist1_ref->Draw("same");
	else hist1_ref->DrawNormalized("same", hist1->GetEntries());
      }
    }
  }
  canvas->Update();
  std::string command = "rm -f " + file_name;
  gSystem->Exec(command.c_str());
  canvas->Print(file_name.c_str(),"png");
  canvas->Clear();
  delete canvas;	
}
//
// -- Set New CondDB Plot
//
void SiStripHistoPlotter::setNewCondDBPlot(std::string& path, std::string& option, int width, int height) {
  PlotParameter local_par;
  local_par.Path    = path;
  local_par.Option  = option;
  local_par.CWidth  = width;
  local_par.CHeight = height;
  condDBPlotList_.push_back(local_par);  
}
//
// -- Create CondDB Plots 
//
void SiStripHistoPlotter::createCondDBPlots(DQMStore* dqm_store) {
  if (condDBPlotList_.size() == 0) return;
  std::string name = "Dummy";
  if (!hasNamedImage(name)) createDummyImage(name);

  for (std::vector<PlotParameter>::iterator it = condDBPlotList_.begin(); 
       it != condDBPlotList_.end(); it++) {
    makeCondDBPlots(dqm_store, (*it));
  }
  condDBPlotList_.clear();
}
//
// -- Draw CondDB Histograms 
//
void SiStripHistoPlotter::makeCondDBPlots(DQMStore* dqm_store, const PlotParameter& par) {
  TCanvas * canvas = new TCanvas("TKCanvas", "TKCanvas", par.CWidth, par.CHeight);

  std::vector<std::string> htypes;
  std::string option = par.Option;
  SiStripUtility::split(option, htypes, ",");

  std::string tag;
  std::vector<MonitorElement*> all_mes = dqm_store->getContents(par.Path);

  for (std::vector<std::string>::const_iterator ih = htypes.begin();
       ih!= htypes.end(); ih++) {
    std::string type = (*ih);
    if (type.size() == 0) continue;
    std::string tag = par.Path + "/";
    for (std::vector<MonitorElement *>::const_iterator it = all_mes.begin();
	 it!= all_mes.end(); it++) {  
      MonitorElement * me = (*it);
      if (!me) continue;
      std::string hname = me->getName();
      if (hname.find(type) != std::string::npos) {
	TH1* histo = me->getTH1();
	histo->Draw();
        tag += type;
	fillNamedImageBuffer(canvas, tag);
	canvas->Clear();        
      } 
    }
  }
  delete canvas;
}

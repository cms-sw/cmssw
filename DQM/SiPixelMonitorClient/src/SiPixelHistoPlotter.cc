#include "DQM/SiPixelMonitorClient/interface/SiPixelHistoPlotter.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
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
#include "TPaveLabel.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include <iostream>
using namespace std;
//
// -- Constructor
// 
SiPixelHistoPlotter::SiPixelHistoPlotter() {
  edm::LogInfo("SiPixelHistoPlotter") << 
    " Creating SiPixelHistoPlotter " << "\n" ;
}
//
// --  Destructor
// 
SiPixelHistoPlotter::~SiPixelHistoPlotter() {
  edm::LogInfo("SiPixelHistoPlotter") << 
    " Deleting SiPixelHistoPlotter " << "\n" ;
  plotList_.clear();

}

//
// -- Set New
//
void SiPixelHistoPlotter::setNewPlot(std::string& path, 
                                     std::string& option, 
				     int width,
				     int height) {
//cout<<"Entering SiPixelHistoPlotter::setNewPlot for "<<path<<endl;
  PlotParameter local_par;
  local_par.Path    = path;
  local_par.Option  = option;
  local_par.CWidth  = width;
  local_par.CHeight = height;
  plotList_.push_back(local_par);  
//cout<<"... leaving SiPixelHistoPlotter::setNewPlot!"<<endl;
}

//
// -- Create Plots 
//
void SiPixelHistoPlotter::createPlots(DQMStore* bei) {
  string name = "Dummy";
  if (!hasNamedImage(name)) createDummyImage(name);
  //cout<<"HP::createPlots:PlotList size="<<plotList_.size()<<endl;
  for (vector<PlotParameter>::iterator it = plotList_.begin(); 
       it != plotList_.end(); it++) {
    makePlot(bei, (*it));
  }
  plotList_.clear();
}

//
// -- Draw Histograms 
//
void SiPixelHistoPlotter::makePlot(DQMStore* bei, const PlotParameter& par) {
//cout<<"Entering SiPixelHistoPlotter::makePlot: "<<endl;
  TCanvas * canvas = new TCanvas("PXCanvas", "PXCanvas", par.CWidth, par.CHeight);
  gROOT->Reset(); 
  gStyle->SetPalette(1,0);
  gStyle->SetOptStat(111110);
  canvas->SetBorderMode(0);
  canvas->SetFillColor(0);

  TPaveText * paveOnCanvas = new TPaveText(0.57,0.79,0.77,0.99,"NDCtr");   
  paveOnCanvas->SetFillColor(0);
  
  //std::cout<<"::makePlot:Trying to plot "<<par.Path<<std::endl;
  MonitorElement * me = bei->get(par.Path);
  if (me) { 
	
    string opt = "";
    if (me->kind() == MonitorElement::DQM_KIND_TH2F) 
      { opt = "COLZ"; gStyle->SetOptStat(10); }
    me->getRootObject()->Draw(opt.c_str());
    int istat =  SiPixelUtility::getStatus(me);
    
    string dopt = par.Option;
    string tag;
    int icol;
    SiPixelUtility::getStatusColor(istat, icol, tag);

    TH1F* histo = dynamic_cast<TH1F*>(me->getRootObject());
    string var = (me->getName()).substr((me->getName()).find_last_of("/")+1);
    if (histo) {
      //setDrawingOption(histo);
      //histo->Draw(opt.c_str());
      string name = histo->GetName();
      if (me->getRefRootObject()) {
        TH1* histo_ref = me->getRefTH1();
        if (histo_ref) {
	  histo_ref->SetLineColor(4);
	  histo_ref->SetMarkerColor(4);
	  if (name.find("SUM") != string::npos) histo_ref->Draw("same");
	  else histo_ref->DrawNormalized("same", histo->GetEntries());
        }
      }
    }
    canvas->Update();
    
    //TH1* histo = me->getTH1();    
    //TH1F* tproject = 0;
    //if (dopt == "projection") {
      //getProjection(me, tproject);
      //if (tproject) tproject->Draw();
      //else histo->Draw();
    //} else {
     // histo->Draw();
    //}
    //TText tTitle;
    //tTitle.SetTextFont(64);
    //tTitle.SetTextSizePixels(20);
    //tTitle.DrawTextNDC(0.1, 0.92, histo->GetName());

    //if (icol != 1) {
    //  TText tt;
    //  tt.SetTextSize(0.12);
    //  tt.SetTextColor(icol);
    //  tt.DrawTextNDC(0.5, 0.5, tag.c_str());
    //}
    
    TLine* l_min = new TLine();
    TLine* l_max = new TLine();
    l_min->SetLineWidth(4);
    l_max->SetLineWidth(4);
  
    
    if(istat!=0){
      TText* statusOnCanvas = paveOnCanvas->AddText(tag.c_str());
      statusOnCanvas->SetTextSize(0.08);
      statusOnCanvas->SetTextFont(112);
      statusOnCanvas->SetNDC(kTRUE);
      statusOnCanvas->SetTextColor(icol);
      
      double ymax = -1.;  
      double ymin = -1.;
      double xmax = -1.;  
      double xmin = -1.;
      double warning = -1.;
      double error = -1.;
      double channelFraction = -1.;
      //if(var.find("SUM") != string::npos){
      //cout << "ME name: " << var << endl;
      setLines(me,var,ymin,ymax,warning,error,channelFraction);
      //cout << "ymin: " << ymin << " ymax: " << ymax << " warning: " << warning << " error: " << error << " channelFraction: " << channelFraction << endl;
	
      if(istat!=dqm::qstatus::STATUS_OK){
        string textMessage = "fraction of channels failing:";
        TText* messageOnCanvas = paveOnCanvas->AddText(textMessage.c_str());
        messageOnCanvas->SetTextSize(0.03);
        messageOnCanvas->SetNDC(kTRUE);
        char text[10];
        sprintf(text,"%.2f %%",channelFraction);
        messageOnCanvas = paveOnCanvas->AddText(text);
        messageOnCanvas->SetTextSize(0.035);
        messageOnCanvas->SetNDC(kTRUE);
      }
      if(ymin!= -1. && ymax!=-1.){
        SiPixelUtility::setDrawingOption(histo);
        l_min->SetLineColor(icol);
        l_max->SetLineColor(icol);
        if(var.find("SUM") != string::npos){	  
	  xmin = histo->GetXaxis()->GetXmin();
	  xmax = histo->GetXaxis()->GetXmax(); 
	  //cout<<"xmin="<<xmin<<" , xmax="<<xmax<<" , ymin="<<ymin<<" , ymax="<<ymax<<endl;
	  l_min->SetX1(xmin);
	  l_min->SetX2(xmax);
	  l_min->SetY1(ymin);
	  l_min->SetY2(ymin);
	  l_min->Draw("same");
	  l_max->SetX1(xmin);
	  l_max->SetX2(xmax);
	  l_max->SetY1(ymax);
	  l_max->SetY2(ymax);
	  l_max->Draw("same");
        }else{
	  xmin = ymin;
	  xmax = ymax;
	  ymin = histo->GetYaxis()->GetBinLowEdge(1);
	  ymax = histo->GetMaximum();
	  //cout<<"xmin="<<xmin<<" , xmax="<<xmax<<" , ymin="<<ymin<<" , ymax="<<ymax<<endl;
	  l_min->SetX1(xmin);
	  l_min->SetX2(xmin);
	  l_min->SetY1(ymin);
	  l_min->SetY2(ymax);
	  l_min->Draw("same");
	  l_max->SetX1(xmax);
	  l_max->SetX2(xmax);
	  l_max->SetY1(ymin);
	  l_max->SetY2(ymax);
	  l_max->Draw("same");
        }
      }
      //setSubDetAxisDrawing((me->getName()),histo);
      //}
      paveOnCanvas->Draw("same");
    }
    if(((var.find("Barrel") != string::npos) && (var.find("SUM") != string::npos)) ||
       ((var.find("Endcap") != string::npos) && (var.find("SUM") != string::npos))) 
      setSubDetAxisDrawing((me->getName()),histo);
    
    canvas->Update();
    
    
    //cout<<"Calling fillNamedImageBuffer now!"<<endl;
    fillNamedImageBuffer(canvas, par.Path);
    canvas->Clear();
  } else {
    createDummyImage(par.Path);
  }
  delete canvas;
  //cout<<"... leaving SiPixelHistoPlotter::makePlot!"<<endl;
}

//
// -- Get Named Image buffer
//
void SiPixelHistoPlotter::getNamedImageBuffer(const string& path, string& image) {
  map<string, string>::iterator cPos = namedPictureBuffer_.find(path);
  if (cPos != namedPictureBuffer_.end()) {
    image = cPos->second;
    if (namedPictureBuffer_.size() > 99 ) namedPictureBuffer_.erase(cPos);
  } else {
     cPos = namedPictureBuffer_.find("Dummy");
     image = cPos->second;
  }
}


/*! \brief (Documentation under construction).
 *
 *  This method 
 */
void SiPixelHistoPlotter::fillNamedImageBuffer(TCanvas * c1, const std::string& name) 
{
//cout<<"Entering SiPixelHistoPlotter::fillNamedImageBuffer: "<<endl;
  //  DQMScope enter;
 // Now extract the image
  // 114 - stands for "no write on Close"
//   cout << ACYellow << ACBold
//        << "[SiPixelInformationExtractor::fillNamedImageBuffer()] "
//        << ACPlain
//        << "A canvas: "
//        << c1->GetName() 
//        << endl ;
  c1->Update();
  c1->Modified();  
  TImageDump imgdump("tmp.png", 114);
  c1->Paint();

// get an internal image which will be automatically deleted
// in the imgdump destructor
  TImage *image = imgdump.GetImage();

  if( image == NULL )
  {
   //cout << "No TImage found for "
//	<< name
   //	<< endl ;
    return ;
  }
  //cout<<"found an image!"<<endl;
  char *buf;
  int sz = 0;
  image->GetImageBuffer(&buf, &sz);

  ostringstream local_str;
  for (int i = 0; i < sz; i++) local_str << buf[i];
  
//  delete [] buf;
  ::free(buf); // buf is allocated via realloc() by a C language AfterStep library invoked by the
               // default (and so far only) TImage implementation in root, TASImage.
  
  // clear the first element map if # of entries > 30
  if (hasNamedImage(name)) namedPictureBuffer_.erase(name);
  //cout<<"filling namedPictureBuffer_["<<name<<"] now"<<endl;
  namedPictureBuffer_[name] = local_str.str();
  //if (namedPictureBuffer_[name].size() > 0) cout << "image created " << name << endl;
//cout<<"... leaving SiPixelHistoPlotter::fillNamedImageBuffer!"<<endl;
}

//
// -- Check if the image exists
//
bool SiPixelHistoPlotter::hasNamedImage(const string& name) {
  map<string, string>::const_iterator cPos = namedPictureBuffer_.find(name);
  if (cPos == namedPictureBuffer_.end()) { 
    return false;
  } else return true;
}

//
// -- Create Dummy Image 
//
void SiPixelHistoPlotter::createDummyImage(const string & name) {
  string          line;
  ostringstream   local_str;
  // Read back the file line by line and temporarily store it in a stringstream
  ifstream * imagefile = new ifstream("images/EmptyPlot.png",ios::in);
  if(imagefile->is_open()) {
    while (getline( *imagefile, line )) {
      local_str << line << endl ;
    }
  }  
  namedPictureBuffer_.insert(pair<string, string>(name, local_str.str()));

  imagefile->close() ;
}

// -- Set Drawing Option
//
void SiPixelHistoPlotter::setDrawingOption(TH1* hist) {
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

//
// -- create static plots
// 
void SiPixelHistoPlotter::createStaticPlot(MonitorElement* me, const string& file_name) {
  TH1* hist1 = me->getTH1();
  TCanvas* canvas  = new TCanvas("TKCanvas", "TKCanvas", 600, 400);
  if (hist1) {
    TText tTitle;
    tTitle.SetTextFont(64);
    tTitle.SetTextSizePixels(20);
    
    setDrawingOption(hist1);
    hist1->Draw();
    string name = hist1->GetName();
    if (me->getRefRootObject()) {
      TH1* hist1_ref = me->getRefTH1();
      if (hist1_ref) {
	hist1_ref->SetLineColor(4);
	hist1_ref->SetMarkerColor(4);
	if (name.find("SUM") != string::npos) hist1_ref->Draw("same");
	else hist1_ref->DrawNormalized("same", hist1->GetEntries());
      }
    }
  }
  canvas->Update();
  string command = "rm -f " + file_name;
  gSystem->Exec(command.c_str());
  canvas->Print(file_name.c_str(),"png");
  canvas->Clear();
  delete canvas;	
}


void SiPixelHistoPlotter::setLines(MonitorElement * me,
					   string & meName, 
					   double & ymin,
					   double & ymax, 
					   double & warning, 
					   double & error, 
					   double & channelFraction) {
//cout<<"Entering SiPixelHistoPlotter::setLines for "<<meName<<endl;
   std::vector<QReport *> report;
   std::string colour;

   if (me->hasError()){
     colour="red";
     report= me->getQErrors();
   } else if( me->hasWarning()){ 
     colour="orange";
     report= me->getQWarnings();
   } else if(me->hasOtherReport()){
     colour="black";
     report= me->getQOthers();
   } else {
     colour="green";
   }
   for(std::vector<QReport *>::iterator itr=report.begin(); itr!=report.end();++itr ){
     std::string text= (*itr)->getMessage();
     //std::cout<<"ME: "<<meName<<" QTmessage:"<<text<<std::endl;
     int num1 = text.find_first_of(":")+1;
     int num2 = text.find_first_of("-",num1)+1;
     int num3 = text.find_first_of(" ",num2);
     std::stringstream dummy(text.substr(num1,num2-num1-1));
     dummy >> ymin; 
     std::stringstream dummy1(text.substr(num2,num3-num2));
     dummy1 >> ymax; 
     //std::cout<<",ymin="<<ymin<<",ymax="<<ymax<<std::endl;
     int num4 = text.find_last_of("=")+2;
     std::stringstream dummy2(text.substr(num4));
     dummy2 >> channelFraction;
     channelFraction = (1.- channelFraction)*100.;
     error = 25.;
     warning = 10.;
     //std::cout<<",warning="<<warning<<",error="<<error<<std::endl;
   }
	  
}

//
// -- Set Axis Drawing Option for slide show plots
//
void SiPixelHistoPlotter::setSubDetAxisDrawing(string detector, TH1F * histo) {

  histo->GetXaxis()->SetLabelColor(0);

  string ownName = "";
  if(detector.find("Barrel") != string::npos || detector.find("Endcap") != string::npos){ 
    if(detector.find("Barrel") != string::npos){
      ownName      = "Shell";
    }
    if(detector.find("Endcap") != string::npos){
      ownName = "HalfCylinder";
    }
    
    TText tt;
    tt.SetTextSize(0.04);
    string mI = ownName + "_mI"; tt.DrawTextNDC(0.12, 0.04, mI.c_str());
    string mO = ownName + "_mO"; tt.DrawTextNDC(0.32, 0.04, mO.c_str());
    string pI = ownName + "_pI"; tt.DrawTextNDC(0.52, 0.04, pI.c_str());
    string pO = ownName + "_pO"; tt.DrawTextNDC(0.72, 0.04, pO.c_str());
  }
  if(detector.find("Shell") != string::npos){
    ownName = "Layer";
    TText tt;
    tt.SetTextSize(0.04);
    string l_one   = ownName + "_1"; tt.DrawTextNDC(0.12, 0.03, l_one.c_str());
    string l_two   = ownName + "_2"; tt.DrawTextNDC(0.42, 0.03, l_two.c_str());
    string l_three = ownName + "_3"; tt.DrawTextNDC(0.72, 0.03, l_three.c_str());
  }
  if(detector.find("HalfCylinder") != string::npos){
    ownName = "Disk";
    TText tt;
    tt.SetTextSize(0.04);
    string d_one   = ownName + "_1"; tt.DrawTextNDC(0.12, 0.03, d_one.c_str());
    string d_two   = ownName + "_2"; tt.DrawTextNDC(0.62, 0.03, d_two.c_str());
  }
}

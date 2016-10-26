#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <TROOT.h>
#include <TStyle.h>
#include "TFile.h"
#include "TH1D.h"
#include "TDirectoryFile.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TLegend.h"
#include "TRint.h"

int main(int argc, char *argv[]) {

  char* filelist;
  char* modulelist;

  filelist= argv[1];
  modulelist= argv[2];

  std::string detid;
  std::string hn;

  std::ifstream inmodules(modulelist);

  while(1){
    
    inmodules >> detid;
    if (!inmodules.good()) break;

    hn="ClusterDigiPosition__det__" + detid;//std::to_string(detid);
    
    TCanvas c1("c1","c1",1600,900);
  
    c1.SetBatch(kTRUE);
    c1.SetLogy(1);
    c1.SetGridy(1);

    TLegend * leg=new TLegend(0.1,0.7,0.2,0.9);
    leg->SetFillStyle(0);

    //cout << detid << endl;
    std::ifstream fileToCountLines(filelist);
    std::size_t lines_count =0;
    std::string line;

    while (std::getline(fileToCountLines , line))
        ++lines_count;

    const float dim=lines_count;

    std::ifstream filesin(filelist);

    std::string filename;
    int k=0;

    TH1D* trend=new TH1D("trend","trend",int(dim),0.5,dim+0.5);
    trend->SetMarkerSize(3);
    trend->SetMarkerStyle(8);
    trend->SetMarkerColor(4);

    std::string ttitle=hn+"_trend";
    trend->SetTitle(ttitle.c_str());

    double max=0;
    double min=1000000;

    while(1){

      filesin >> filename;
      if (!filesin.good()) break;      

      std::string runNum= filename.substr(filename.find("28"), 6);
      std::cout << runNum << std::endl;
     
      TFile * fin=new TFile(filename.c_str()); 
      
      TH1D* Events=(TH1D*)fin->Get("TotEvents");

      double EvtNum=Events->GetBinContent(1);

      TH1D* histo=(TH1D*) fin->Get(hn.c_str());
      

      histo->Scale(1/EvtNum);
      double numberPerEvent= histo->Integral();

      if (max<histo->GetMaximum()) max=histo->GetMaximum();
      if (min>histo->GetMinimum()) min=histo->GetMinimum();
      

      histo->SetLineColor(k+1);
      histo->SetMarkerStyle(9);
      histo->SetMarkerColor(k+1);

      gStyle->SetOptStat(0);
      gROOT->SetStyle("Plain");
      
      if (min==0) min=1.e-6;

      histo->GetYaxis()->SetRangeUser(min,max*2);

      trend->SetBinContent(k+1,numberPerEvent);
      trend->GetXaxis()->SetBinLabel(k+1,runNum.c_str());


      if (k==0){
	histo->Draw("L");
      }

      else histo->Draw("LSAME");
      leg->AddEntry(histo, runNum.c_str(), "L");

      k++;

     
    }

    TLine l;
    
    l.SetLineColor(4);
    l.SetLineStyle(2);
    l.SetLineWidth(3);

    l.DrawLine(128,min,128,max*2);
    l.DrawLine(256,min,256,max*2);
    l.DrawLine(384,min,384,max*2);
    l.DrawLine(384,min,384,max*2);
    l.DrawLine(512,min,512,max*2);
    l.DrawLine(640,min,640,max*2);

    leg->Draw();
    std::string outname= hn+"_Super.png";
    c1.SaveAs(outname.c_str());
    c1.SetLogy(1);
    c1.SetGridx(1);
    trend->GetYaxis()->SetRangeUser(trend->GetMinimum()*0.2,trend->GetMaximum()*2);
    trend->Draw("P");
    outname=hn+"_Trend.png";
    c1.SaveAs(outname.c_str());
    delete leg;
    delete trend;      
   
  }  
  return 0;
}



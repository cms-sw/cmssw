#include "TCanvas.h"
#include "TDirectory.h"
#include "TF1.h"
#include "TFile.h"
#include "TFitResult.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TH1D.h"
#include "TH2D.h"
#include "THStack.h"
#include "TLegend.h"
#include "TMath.h"
#include "TProfile.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TString.h"
#include "TStyle.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>


int         markStyle[7]  = {20, 21, 24, 22, 23, 33, 25};
int         colors[7]     = {1, 2, 4, 6, 7, 38, 3};
int         lineStyle[7]  = {1, 2, 3, 4, 1, 2, 3};

const unsigned int nmodels=2;
//std::string filem[nmodels]={"pikp/FBE4r00MixStudyHLT.root","pikp/FBE4r00vMixStudyHLT.root"};
std::string filem[nmodels]={"pikp/FBE4cMixStudyHLT10.root","pikp/FBE4vcMixStudyHLT10.root"};
std::string typem[nmodels]={"10.4 FTFP_BERT_EMM (Native)","10.4 FTFP_BERT_EMM (VecGeom v0.5)"};

TH1D* getEffi(TFile* file, std::string varname, unsigned int ifl) {
  
  char name[100];
  sprintf(name, "h_%s_All_0", varname.c_str());
  TH1D  *hist1 = (TH1D*) file->FindObjectAny(name);
  sprintf(name, "h_%s_All_1", varname.c_str());
  TH1D  *hist2 = (TH1D*) file->FindObjectAny(name);
  if (hist1 && hist2) {
    sprintf(name, "h_%s_Effy_%d", varname.c_str(), ifl);
    int    nbins = hist1->GetNbinsX();
    double xl    = hist1->GetBinLowEdge(1);
    double xh    = hist1->GetBinLowEdge(nbins) + hist1->GetBinWidth(nbins);
    TH1D* hist = new TH1D(name,hist1->GetTitle(),nbins,xl,xh);
    for (int i=1; i<nbins; ++i) {
      double den = hist1->GetBinContent(i);
      double val = (den > 0) ? (hist2->GetBinContent(i))/den : 0;
      double err = (den > 0) ? (hist1->GetBinError(i))*(val/den) : 0;
      hist->SetBinContent(i,val);
      hist->SetBinError(i,err);
    }
    return hist;
  } else {
    return 0;
  }
}

TCanvas* plotEffi(int type, bool approve) {

  std::string varnam[4] = {"pt","p","eta","phi"};
  std::string xvtitl[4] = {"p_{T} (GeV)", "p (GeV)", "#eta", "#phi"};
  bool        irng[4]   = {true, true, true, false};
  double      xlowr[4]  = { 0.0,  0.0, -2.2, -3.1415926};
  double      xtopr[4]  = {20.0, 20.0,  2.2,  3.1415926};

  TCanvas* c(0);
  if (type < 0 || type > 3) type = 0;
  TObjArray                histArr;
  for (unsigned k=0; k<nmodels; ++k) {
    TFile *file = TFile::Open(filem[k].c_str());
    TH1D  *hist = getEffi(file, varnam[type], k);
    if (hist) {
      hist->GetXaxis()->SetTitle(xvtitl[type].c_str());
      hist->GetYaxis()->SetTitle("Efficiency");
      if (irng[type]) hist->GetXaxis()->SetRangeUser(xlowr[type],xtopr[type]);
      histArr.AddLast(hist);
    }
  }
  if (histArr.GetEntries()>0) {
    gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
    gStyle->SetOptTitle(kFALSE);    gStyle->SetPadBorderMode(0);
    gStyle->SetCanvasBorderMode(0); gStyle->SetOptStat(0);

    char cname[50];
    sprintf (cname, "c_%sEff", varnam[type].c_str());  
    c = new TCanvas(cname, cname, 500, 500);
    gPad->SetTopMargin(0.10);  gPad->SetBottomMargin(0.10);
    gPad->SetLeftMargin(0.15); gPad->SetRightMargin(0.025);

    TLegend  *legend = new TLegend(0.30, 0.15, 0.975, 0.30);
    TPaveText *text1 = new TPaveText(0.05, 0.94, 0.35, 0.99, "brNDC");
    legend->SetBorderSize(1); legend->SetFillColor(kWhite);
    char texts[200];
    sprintf (texts, "CMS Preliminary");
    text1->AddText(texts);
    THStack *Hs      = new THStack("hs2"," ");
    for (int i=0; i<histArr.GetEntries(); i++) {
      TH1D *h =  (TH1D*)histArr[i];
      h->SetLineColor(colors[i]);
      h->SetLineWidth(2);
      h->SetMarkerSize(markStyle[i]);
      Hs->Add(h, "hist sames");
      legend->AddEntry(h,typem[i].c_str(),"l");
    }
    Hs->Draw("nostack");
    c->Update();
    Hs->GetHistogram()->GetXaxis()->SetTitle(xvtitl[type].c_str());
    Hs->GetHistogram()->GetXaxis()->SetLabelSize(0.035);
    Hs->GetHistogram()->GetYaxis()->SetTitleOffset(1.6);
    Hs->GetHistogram()->GetYaxis()->SetTitle("Track Reconstruction Efficiency");
    Hs->GetHistogram()->GetYaxis()->SetRangeUser(0.0,1.2);
    if (irng[type])
      Hs->GetHistogram()->GetXaxis()->SetRangeUser(xlowr[type],xtopr[type]);
    c->Modified();
    c->Update();
    legend->Draw("");
    if (approve) text1->Draw("same");
    c->Modified();
    c->Update();
  }
  return c;
}

void plotEffiAll(bool approve=false, int savePlot=-1) {
  for (int var=0; var<=4; ++var) {
    TCanvas* c = plotEffi(var, approve);
    if (c != 0 && savePlot >= 0 && savePlot < 3) {
      std::string ext[3] = {"eps", "gif", "pdf"};
      char name[200];
      sprintf (name, "%s.%s", c->GetName(), ext[savePlot].c_str());
      c->Print(name);
    }
  }
}

void plotCompare(const char* infile1, const char* text1, const char* infile2,
		 const char* text2, int type1=-1, int type2=-1, int type3=-1,
		 bool logy=true, bool save=false) {

  int         ndets[4]  = {1, 9, 9, 15};
  int         types[4]  = {7, 8, 2, 3};
  std::string htype0[7] = {"PtInc", "EneInc", "EtaInc", "PhiInc", 
			   "HitLow", "HitHigh", "HitMu"};
  int         rebin0[7] = {1, 1, 1, 1, 10, 10, 10};
  std::string htype1[8] = {"Hit", "Time", "Edep", "Etot", "TimeAll", "EdepEM", 
			   "EdepHad", "EtotG"};
  int         rebin1[8] = {10, 10, 100, 100, 10, 50, 50, 50};
  std::string htype2[2] = {"EdepCal", "EdepCalT"};
  int         rebin2[2] = {50, 50};
  std::string htype3[3] = {"HitTk", "TimeTk", "EdepTk"};
  int         rebin3[3] = {10, 10, 50};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);         gStyle->SetOptFit(0);
  gStyle->SetOptStat(1110);

  int itmin1 = (type1 >= 0) ? type1 : 0;
  int itmax1 = (type1 >= 0) ? type1 : 3;
  TFile *file1 = new TFile(infile1);
  TFile *file2 = new TFile(infile2);
  std::cout << "File1: " << infile1 << ":" << file1 << " File2: " << infile2
	    << ":" << file2 << std::endl;
  if (file1 != 0 && file2 != 0) {
    for (int it1=itmin1; it1<=itmax1; ++it1) {
      int itmin2 = (type2 >= 0) ? type2 : 0;
      int itmax2 = (type2 >= 0) ? type2 : ((type1 == 1) ? 3 : types[it1]-1);
      int itmin3 = (type3 >= 0) ? type3 : 0;
      int itmax3 = (type3 >= 0) ? type3 : ndets[it1]-1;
      for (int it2=itmin2; it2<=itmax2; ++it2) {
	int rebin(1);
	if      (it1 == 0) rebin = rebin0[it2];
	else if (it1 == 1) rebin = rebin1[it2];
	else if (it1 == 2) rebin = rebin2[it2];
	else if (it1 == 3) rebin = rebin3[it2];
	for (int it3=itmin3; it3<=itmax3; ++it3) {
	  if (type1 == 1 && (it3 == 1 || it3 == 2)) continue;
	  char name[20], namec[22];
	  if      (it1 == 0) sprintf (name, "%s",   htype0[it2].c_str());
	  else if (it1 == 1) sprintf (name, "%s%d", htype1[it2].c_str(), it3);
	  else if (it1 == 2) sprintf (name, "%s%d", htype2[it2].c_str(), it3);
	  else               sprintf (name, "%s%d", htype3[it2].c_str(), it3);
	  TH1D* hist[2];
	  hist[0] = (TH1D*)file1->FindObjectAny(name);
	  hist[1] = (TH1D*)file2->FindObjectAny(name);
	  if (hist[0] != 0 && hist[1] != 0) {
	    sprintf (namec, "c_%s", name);
	    TCanvas *pad = new TCanvas(namec, namec, 500, 500);
	    pad->SetRightMargin(0.10);
	    pad->SetTopMargin(0.10);
	    if (logy) pad->SetLogy();
	    TLegend *legend = new TLegend(0.12, 0.79, 0.64, 0.89);
	    legend->SetFillColor(kWhite);
	    double ymax(0.90), dy(0.08);
	    for (int ih=0; ih<2; ++ih) {
	      hist[ih]->GetXaxis()->SetTitle(hist[ih]->GetTitle());
	      hist[ih]->SetMarkerStyle(markStyle[ih]);
	      hist[ih]->SetMarkerColor(colors[ih]);
	      hist[ih]->SetLineStyle(lineStyle[ih]);
	      hist[ih]->SetLineColor(colors[ih]);
	      hist[ih]->SetLineWidth(2);
	      hist[ih]->GetYaxis()->SetTitleOffset(1.20);
	      if (rebin > 1) hist[ih]->Rebin(rebin);
	      if (ih == 0) {
		legend->AddEntry(hist[ih],text1,"lp");
		hist[ih]->Draw();
	      } else {
		legend->AddEntry(hist[ih],text2,"lp");
		hist[ih]->Draw("sames");
	      }
	      pad->Update();
	      TPaveStats* st1 = (TPaveStats*)hist[ih]->GetListOfFunctions()->FindObject("stats");
	      if (st1 != NULL) {
		st1->SetLineColor(colors[ih]);
		st1->SetTextColor(colors[ih]);
		st1->SetY1NDC(ymax-dy); st1->SetY2NDC(ymax);
		st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
		ymax -= dy;
	      }
	      pad->Update();
	    }
	    legend->Draw("same");
	    pad->Update();
	    if (save) {
	      sprintf (name, "%s.pdf", pad->GetName());
	      pad->Print(name);
	    }	
	  }
	}
      }
    }
  }
}

void plotCompareAll(const std::string & cdir1="10.4.r00.g4",
		    const std::string & cdir2="10.4.r00.vg",
		    const std::string & cvers="10.4 MinBias",
		    const std::string & cfile= "minbias.root",
		    const std::string & ctype1="Native", 
		    const std::string & ctype2="VecGeom v0.5",
		    bool logy=true, bool save=false) {

  char infile1[200], infile2[200], text1[200], text2[200];
  sprintf (infile1, "%s/%s", cdir1.c_str(), cfile.c_str());
  sprintf (infile2, "%s/%s", cdir2.c_str(), cfile.c_str());
  sprintf (text1,   "%s (%s)", cvers.c_str(), ctype1.c_str());
  sprintf (text2,   "%s (%s)", cvers.c_str(), ctype2.c_str());
  plotCompare(infile1,text1,infile2,text2,1,-1,0,logy,save);
  plotCompare(infile1,text1,infile2,text2,1,-1,3,logy,save);
  plotCompare(infile1,text1,infile2,text2,1,-1,4,logy,save);
  plotCompare(infile1,text1,infile2,text2,1,-1,5,logy,save);
  plotCompare(infile1,text1,infile2,text2,1,-1,6,logy,save);
  plotCompare(infile1,text1,infile2,text2,1,-1,7,logy,save);
  plotCompare(infile1,text1,infile2,text2,1,-1,8,logy,save);
}

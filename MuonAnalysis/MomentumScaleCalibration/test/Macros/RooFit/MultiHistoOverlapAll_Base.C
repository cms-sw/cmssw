#include <string>
#include <vector>
#include "TF1.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TCutG.h"
#include "TFile.h"
#include "TLegend.h"
#include "TMath.h"
#include "TNtuple.h"
#include "TPad.h"
#include "TPaveText.h"
#include "TROOT.h"
#include "TString.h"
#include "TSystem.h"
#include <MuonAnalysis/MomentumScaleCalibration/test/Macros/RooFit/TkAlStyle.cc>


using namespace ROOT::Math;

struct Plot_1D{
  TString directory;
  TString histoname;
  TString plotname;
  TString xtitle;
  TString ytitle;
  float xmin;
  float xmax;
  float ymin;
  float ymax;
  float absMin;
  float absMax;

  vector<TH1D*> histogram;
  vector<TString> histogram_label;
  vector<pair<float, float>> mean_sigma_Y;
  vector<pair<int, float>> chi2Y;

  Plot_1D(TString directory_, TString histoname_, TString plotname_, TString xtitle_, TString ytitle_, float xmin_, float xmax_, float ymin_, float ymax_) :
    directory(directory_),
    histoname(histoname_),
    plotname(plotname_),
    xtitle(xtitle_),
    ytitle(ytitle_),
    xmin(xmin_),
    xmax(xmax_),
    ymin(ymin_),
    ymax(ymax_),
    absMin(9e9),
    absMax(-9e9)
  {}
  ~Plot_1D(){}

  void addMeanSigmaChi2Y(TH1D* htmp){
    double mean=0;
    double sigma=0;
    for (int bin=1; bin<=htmp->GetNbinsX(); bin++){
      float bincontent = htmp->GetBinContent(bin);
      float binerror = htmp->GetBinError(bin);
      if (binerror==0. && bincontent==0.) continue;
      absMin = min(absMin, bincontent - binerror);
      absMax = max(absMax, bincontent + binerror);
      mean += bincontent/pow(binerror, 2);
      sigma += 1./pow(binerror, 2);
    }
    mean /= sigma;
    sigma = sqrt(1./sigma);
    mean_sigma_Y.push_back(pair<float, float>(mean, sigma));

    float chi2=0; int ndof=-1;
    for (int bin=1; bin<=htmp->GetNbinsX(); bin++){
      float bincontent = htmp->GetBinContent(bin);
      float binerror = htmp->GetBinError(bin);
      if (binerror==0. && bincontent==0.) continue;
      ndof++;
      chi2 += pow((bincontent-mean)/binerror, 2);
    }
    chi2Y.push_back(pair<int, float>(ndof, chi2));
  }

  void addHistogramFromFile(TFile* finput, TString label, int color, int linestyle, int markerstyle){
    if (finput!=0 && !finput->IsZombie()){
      TH1D* htmp = (TH1D*)finput->Get(histoname);
      if (htmp!=0){
        htmp->SetTitle("");
        htmp->GetXaxis()->SetTitle(xtitle);
        htmp->GetYaxis()->SetTitle(ytitle);

        htmp->SetLineWidth(1);
        htmp->SetLineColor(color);
        htmp->SetLineStyle(linestyle);
        htmp->SetMarkerSize(1.2);
        htmp->SetMarkerColor(color);
        htmp->SetMarkerStyle(markerstyle);

        histogram.push_back(htmp);
        histogram_label.push_back(label);
        addMeanSigmaChi2Y(htmp);
      }
    }
  }

  void plot(TString fitFormula="", bool AutoSetRange=false){
    // Determine the plot ranges
    unsigned int nValidHistos = histogram.size();
    const unsigned int nValidHistosLimit = 4;
    float rangeFactor[2]={ 1.01, 1.05 };
    float dampingFactor = 0;
    float deviationThreshold = 1.04;
    if (nValidHistos>nValidHistosLimit) dampingFactor = 0.07/nValidHistosLimit*(nValidHistos-nValidHistosLimit);
    double rangeMaxReduction = 0.02;
    if (nValidHistos>nValidHistosLimit) rangeMaxReduction = rangeMaxReduction*nValidHistosLimit/nValidHistos;
    double dampingFactorEff = dampingFactor;

    float overallAvg=0;
    float overallSigmaSqInv=0;
    for (unsigned int f=0; f<nValidHistos; f++){
      float avgY = mean_sigma_Y.at(f).first;
      float devY = mean_sigma_Y.at(f).second;
      devY = 1./pow(devY, 2);
      avgY *= devY;
      overallAvg += avgY;
      overallSigmaSqInv += devY;
    }
    overallAvg /= overallSigmaSqInv;

    // Update August 2018, implemented autorange for histograms
    float MaxY=-1000.;
    float MinY=1000.;


    for (unsigned int f=0; f<nValidHistos; f++){
      for (int bin=1; bin<=histogram.at(f)->GetNbinsX(); bin++){
        float bincontent = histogram.at(f)->GetBinContent(bin);
        float binerror = histogram.at(f)->GetBinError(bin);
        if (binerror==0 && bincontent==0) continue;
        if ((bincontent + binerror)>deviationThreshold*overallAvg) rangeMaxReduction = 0;
	if(bincontent>=MaxY)MaxY=bincontent;
	if(bincontent<=MinY)MinY=bincontent;

      }
    }
    if (nValidHistos>nValidHistosLimit && rangeMaxReduction!=0) dampingFactorEff = dampingFactorEff*0.7;
    
    float Yrange = MaxY-MinY; 
    if(AutoSetRange){ // Overwtite fixed range for Y axis with the one computed for the current histogram
      ymin=MinY-Yrange/10.;
      ymax=Yrange*1.6+ymin;
    }

   
    if (ymin>=ymax){ // If range is not already set
      ymin = absMin/rangeFactor[0];
      ymax = absMax*(rangeFactor[1]+dampingFactorEff-rangeMaxReduction);
    }
    for (unsigned int f=0; f<nValidHistos; f++){
      histogram.at(f)->GetXaxis()->SetRangeUser(xmin, xmax);
      histogram.at(f)->GetYaxis()->SetRangeUser(ymin, ymax);
    }

    // Begin plotting
    vector<TF1*> hfit;
    TCanvas* canvas;
    TLegend* legend;
    TString cname = Form("c%s", plotname.Data());
    canvas = new TCanvas(cname, cname, 8, 30, 800, 800);
    canvas->cd();
    legend = TkAlStyle::legend("topleft", nValidHistos);
    for (unsigned int f=0; f<nValidHistos; f++){
      TString legendlabel = histogram_label.at(f);
      if (f==0) histogram.at(f)->Draw();
      else histogram.at(f)->Draw("same");

      if (fitFormula==""){
        if (plotname.Contains("mean")){
          float normchi2=0.;
          if (chi2Y.at(f).first>0) normchi2=chi2Y.at(f).first/chi2Y.at(f).second;
          legendlabel = Form("%s (#chi^{2}/dof=%.2f)", legendlabel.Data(), normchi2);
        }
      }
      else{
        TF1* ftmp = new TF1(Form("fit_%i", f), fitFormula, xmin, xmax);
        ftmp->SetParameter(0, 91.0);
        ftmp->SetParameter(1, 0);
        if (fitFormula.Contains("[2]")) ftmp->SetParameter(2, 0);
        ftmp->SetLineColor(1);

        histogram.at(f)->Fit(ftmp, "R", "same", xmin, xmax);
        ftmp->Draw("same");
        hfit.push_back(ftmp);
      }

      legend->AddEntry(histogram.at(f), legendlabel, "lp");
    }
    legend->Draw("same");
    TkAlStyle::drawStandardTitle();
    canvas->RedrawAxis();
    canvas->Modified();
    canvas->Update();
    canvas->SaveAs(Form("%s/%s%s", directory.Data(), plotname.Data(), ".png"));
    canvas->SaveAs(Form("%s/%s%s", directory.Data(), plotname.Data(), ".pdf"));

    for (unsigned int hf=0; hf<hfit.size(); hf++) delete hfit.at(hf);
    delete legend;
    canvas->Close();
    delete canvas;
  }

};


void splitOption(string rawoption, string& wish, string& value, char delimiter){
  size_t posEq = rawoption.find(delimiter);
  if (posEq!=string::npos){
    wish=rawoption;
    value=rawoption.substr(posEq+1);
    wish.erase(wish.begin()+posEq, wish.end());
  }
  else{
    wish="";
    value=rawoption;
  }
}
void splitOptionRecursive(string rawoption, vector<string>& splitoptions, char delimiter){
  string suboption=rawoption, result=rawoption;
  string remnant;
  while (result!=""){
    splitOption(suboption, result, remnant, delimiter);
    if (result!="") splitoptions.push_back(result);
    suboption = remnant;
  }
  if (remnant!="") splitoptions.push_back(remnant);
}

void getCustomRanges(TString type, TString resonance, int iP, float minmax_plot[2]);
void MultiHistoOverlapAll_Base_one(string files, string labels, string colors, string linestyles, string markerstyles, TString directory, TString resonance, TString type, bool switchONfit, bool AutoSetRange=false, float CustomMinY=90.85, float CustomMaxY=91.4);

void MultiHistoOverlapAll_Base(string files, string labels, string colors, string linestyles, string markerstyles, TString directory, TString resonance, bool switchONfit=false, bool AutoSetRange=false, float CustomMinY=90.85, float CustomMaxY=91.4){
  gSystem->mkdir(directory, true);
  MultiHistoOverlapAll_Base_one(files, labels, colors, linestyles, markerstyles, directory, resonance, "mean", switchONfit, AutoSetRange, CustomMinY, CustomMaxY);
  MultiHistoOverlapAll_Base_one(files, labels, colors, linestyles, markerstyles, directory, resonance, "sigma", switchONfit, AutoSetRange, CustomMinY, CustomMaxY);
}

void MultiHistoOverlapAll_Base_one(string files, string labels, string colors, string linestyles, string markerstyles, TString directory, TString resonance, TString type, bool switchONfit, bool AutoSetRange, float CustomMinY, float CustomMaxY){
  gROOT->Reset();
  if (TkAlStyle::status() == NO_STATUS) TkAlStyle::set(INTERNAL);
  gROOT->ForceStyle();

  vector<string> strValidation_file;
  vector<string> strValidation_label;
  vector<string> strValidation_color;
  vector<string> strValidation_linestyle;
  vector<string> strValidation_markerstyle;
  splitOptionRecursive(files, strValidation_file, ',');
  splitOptionRecursive(labels, strValidation_label, ',');
  splitOptionRecursive(colors, strValidation_color, ',');
  splitOptionRecursive(linestyles, strValidation_linestyle, ',');
  splitOptionRecursive(markerstyles, strValidation_markerstyle, ',');
  int nfiles = strValidation_file.size();
  int nlabels = strValidation_label.size();
  int ncolors = strValidation_color.size();
  int nlinestyles = strValidation_linestyle.size();
  int nmarkerstyles = strValidation_markerstyle.size();
  if (nlabels!=nfiles){
    cout << "nlabels!=nfiles" << endl;
    return;
  }
  if (ncolors!=0 && ncolors!=nfiles){
    cout << "ncolors!=nfiles" << endl;
    return;
  }
  if (nlinestyles!=0 && nlinestyles!=nfiles){
    cout << "nlinestyles!=nfiles" << endl;
    return;
  }
  if (nmarkerstyles!=0 && nmarkerstyles!=nfiles){
    cout << "nmarkerstyles!=nfiles" << endl;
    return;
  }

  const int nhistos=8;
  const int n2Dhistos = 2;
  TFile** file = new TFile*[nfiles];
  for (int f=0; f<nfiles; f++) file[f] = TFile::Open((strValidation_file[f]).c_str(), "read");

  TString histoname[nhistos] ={
    TString("MassVsPt/allHistos/") + type + TString("Histo"),
    TString("MassVsPhiPlus/allHistos/") + type + TString("Histo"),
    TString("MassVsPhiMinus/allHistos/") + type + TString("Histo"),
    TString("MassVsEtaPlus/allHistos/") + type + TString("Histo"),
    TString("MassVsEtaMinus/allHistos/") + type + TString("Histo"),
    TString("MassVsEtaPlusMinusDiff/allHistos/") + type + TString("Histo"),
    TString("MassVsCosThetaCS/allHistos/") + type + TString("Histo"),
    TString("MassVsPhiCS/allHistos/") + type + TString("Histo")
  };
  TString histo2Dname[n2Dhistos] ={
    TString("MassVsEtaPhiPlus/allHistos/") + type + TString("Histo"),
    TString("MassVsEtaPhiMinus/allHistos/") + type + TString("Histo"),
  };
  TString xtitle[nhistos] ={
    "p^{T}_{#mu} (GeV)",
    "#phi_{#mu+}",
    "#phi_{#mu-}",
    "#eta_{#mu+}",
    "#eta_{#mu-}",
    "#eta_{#mu+} - #eta_{#mu-}",
    "cos #theta_{CS}",
    "#phi_{CS}"
  };
  TString x2Dtitle[n2Dhistos] ={
    "#phi_{#mu+}",
    "#phi_{#mu-}"
  };
  TString y2Dtitle[n2Dhistos] ={
    "#eta_{#mu+}",
    "#eta_{#mu-}"
  };
  TString ytitle;
  if (type=="mean") ytitle = "M_{#mu#mu} (GeV)";
  else if (type=="sigma") ytitle = "#sigma_{#mu#mu} (GeV)";
  TString plotname[nhistos] ={
    type + "MassVsPt_ALL",
    type + "MassVsPhiPlus_ALL",
    type + "MassVsPhiMinus_ALL",
    type + "MassVsEtaPlus_ALL",
    type + "MassVsEtaMinus_ALL",
    type + "MassVsDeltaEta_ALL",
    type + "MassVsCosThetaCS_ALL",
    type + "MassVsPhiCS_ALL"
  };
  TString plotname2D[n2Dhistos] ={
    type + "MassVsEtaPhiPlus",
    type + "MassVsEtaPhiMinus"
  };
  TString fitFormula[nhistos]={
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*x",
    "[0]+[1]*x",
    "[0]+[1]*x",
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*cos(x+[2])"
  };
  float plot_xmax[nhistos]={
    static_cast<float>((resonance=="Z" ? 100. : (resonance=="Y1S" ? 20. : (resonance=="Y2S" ? 20. : (resonance=="Y3S" ? 20. : (resonance=="JPsi" ? 20. : 20.)))))),
    static_cast<float>(TMath::Pi()),
    static_cast<float>(TMath::Pi()),
    2.4,
    2.4,
    4.8,
    1,
    static_cast<float>(TMath::Pi())
  };
  float plot_xmin[nhistos]={
    0.,
    static_cast<float>(-TMath::Pi()),
    static_cast<float>(-TMath::Pi()),
    -2.4,
    -2.4,
    -4.8,
    -1,
    static_cast<float>(-TMath::Pi())
  };


  // Plot 2D hystograms
  // TO BE FIXED: something strange appears in the 2D plots for the sigma, this should be further investigated
  Double_t zMass(91.1876);
  Double_t Y1SMass(9.46);
  Double_t maxDist(0.);
  Double_t Mass=zMass;
  Double_t fixrange=1.5;
  if (resonance=="Y1S"){
    Mass=Y1SMass;
    fixrange=0.15;
  }
  for (int f=0; f<nfiles; f++){
 
    if (file[f]!=0 && !file[f]->IsZombie()){

      
      TH2D* histo2D[2];
      histo2D[0]= (TH2D*)file[f]->Get(histo2Dname[0]); //histoMassVsEtaPhiPlus
      histo2D[1]= (TH2D*)file[f]->Get(histo2Dname[1]); //histoMassVsEtaPhiMinus
      for(int s=0;s<2;s++){
	TCanvas dummycanvas;
	dummycanvas.SetFillColor(0);  
	dummycanvas.cd()->SetTopMargin(0.07);
	dummycanvas.cd()->SetRightMargin(0.15);
	dummycanvas.cd()->SetLeftMargin(0.12);
	histo2D[s]->SetTitle("");
	histo2D[s]->GetZaxis()->SetLabelFont(42);
	histo2D[s]->GetXaxis()->SetTitle(x2Dtitle[s]);
	histo2D[s]->GetYaxis()->SetTitle(y2Dtitle[s]);
	histo2D[s]->GetYaxis()->SetTitleOffset(0.9);
	if(type=="mean"){//This is made to ensure that the range options are not used for the sigma
	  Double_t zMin=Mass;
	  Double_t zMax=Mass;
 
	  for(int nbinX=1;nbinX<=histo2D[s]->GetNbinsX();nbinX++){
	    for(int nbinY=1;nbinY<=histo2D[s]->GetNbinsY();nbinY++){
	      Double_t value = histo2D[s]->GetBinContent(nbinX,nbinY);
	      if(value<zMin)zMin=value;
	      if(value>zMax)zMax=value;
	    }
	  }
	  maxDist=fabs(zMax-Mass);
	  if(fabs(Mass-zMin) >= maxDist) maxDist=fabs(Mass-zMin);
	  histo2D[s]->GetZaxis()->SetRangeUser(Mass-fixrange,Mass+fixrange);//Default range Zmass +- 1.5 GeV (Ymass +- 0.15 GeV), NOTE: this will create empty bins when the bin content is either lower or higher than the fixed range
	  if(AutoSetRange)histo2D[s]->GetZaxis()->SetRangeUser(zMin,zMax); //Set range automatically
	  
	}
	histo2D[s]->Draw("COLZ");
	TString alignment_label = strValidation_label.at(f);
	alignment_label.ReplaceAll(" ","_");
	dummycanvas.SaveAs(Form("%s/%sc%s%s%s", directory.Data(),alignment_label.Data(),"_", plotname2D[s].Data(), ".png"));
	dummycanvas.SaveAs(Form("%s/%sc%s%s%s", directory.Data(),alignment_label.Data(),"_", plotname2D[s].Data(), ".pdf"));
      }
      
    }

  };
  

  for (int iP=0; iP<nhistos; iP++){
    TString theFitFormula="";
    if (switchONfit && type=="mean") theFitFormula = fitFormula[iP];

    float minmax_plot[2]={ -1, -1 };
    //getCustomRanges(type, resonance, iP, minmax_plot);
    if (type=="mean") {
      minmax_plot[0]=CustomMinY;
      minmax_plot[1]=CustomMaxY;
    }
    Plot_1D thePlot(directory, histoname[iP], plotname[iP], xtitle[iP], ytitle, plot_xmin[iP], plot_xmax[iP], minmax_plot[0], minmax_plot[1]);
    for (int f=0; f<nfiles; f++){
      int color=0, linestyle=0, markerstyle=0;
      if (ncolors>f) color = stoi(strValidation_color.at(f));
      else if (f==0) color = (int)kBlack;
      else if (f==(nfiles-1)) color = (int)kViolet;
      else if (f==1) color = (int)kBlue;
      else if (f==2) color = (int)kRed;
      else if (f==3) color = (int)(kGreen+2);
      else if (f==4) color = (int)(kOrange+3);
      else if (f==5) color = (int)kGreen;
      else if (f==6) color = (int)kYellow;
      else if (f==7) color = (int)(kPink+9);
      else if (f==8) color = (int)kCyan;
      else if (f==9) color = (int)(kGreen+3);
      else color = (int)kBlack;
      if (nlinestyles>f){
        linestyle = stoi(strValidation_linestyle.at(f));
        if (linestyle > 100) markerstyle = (linestyle / 100);
        linestyle = linestyle % 100;
      }
      else linestyle=1;
      if (nmarkerstyles>f) markerstyle = stoi(strValidation_markerstyle.at(f));
      else if (markerstyle==0){
        if (strValidation_label.at(f).find("reference")!=string::npos || strValidation_label.at(f).find("Reference")!=string::npos) markerstyle=1;
        else markerstyle=20;
      }
      thePlot.addHistogramFromFile(file[f], strValidation_label.at(f), color, linestyle, markerstyle);
    }
    if(type=="mean")thePlot.plot(theFitFormula,AutoSetRange);
    else thePlot.plot(theFitFormula,true);
    
  }
  
  for (int f=nfiles-1; f>=0; f--){ if (file[f]!=0 && file[f]->IsOpen()) file[f]->Close(); }
  delete[] file;
}

void getCustomRanges(TString type, TString resonance, int iP, float minmax_plot[2]){
  if (type=="mean"){ // Custom ranges

    if (resonance=="Z"){
      if (iP==7){ // PhiCS
        minmax_plot[0] = 90.9;
        minmax_plot[1] = 91.3;
      }
      else if (iP==6){ // cosThetaCS
        minmax_plot[0] = 90.95;
        minmax_plot[1] = 91.4;
      }
      else if (iP==5){ // deta
        minmax_plot[0] = 90.9;
        minmax_plot[1] = 91.6;
      }
      else if (iP==4){ // eta-
        minmax_plot[0] = 90.8;
        minmax_plot[1] = 91.5;
      }
      else if (iP==3){ // eta+
        minmax_plot[0] = 90.8;
        minmax_plot[1] = 91.5;
      }
      else if (iP==2){ // phi-
        minmax_plot[0] = 90.8;
        minmax_plot[1] = 91.7;
      }
      else if (iP==1){ // phi+
        minmax_plot[0] = 90.8;
        minmax_plot[1] = 91.7;
      }
      else{ // pt
        minmax_plot[0] = 90.;
        minmax_plot[1] = 94.;
      }
    }
    else if (resonance=="Y1S"){
      if (iP==7){ // PhiCS
        minmax_plot[0] = 9.42;
        minmax_plot[1] = 9.50;
      }
      else if (iP==6){ // cosThetaCS
        minmax_plot[0] = 9.43;
        minmax_plot[1] = 9.49;
      }
      else if (iP==5){ // deta
        minmax_plot[0] = 9.43;
        minmax_plot[1] = 9.49;
      }
      else if (iP==4){ // eta-
        minmax_plot[0] = 9.40;
        minmax_plot[1] = 9.52;
      }
      else if (iP==3){ // eta+
        minmax_plot[0] = 9.41;
        minmax_plot[1] = 9.52;
      }
      else if (iP==2){ // phi-
        minmax_plot[0] = 9.435;
        minmax_plot[1] = 9.475;
      }
      else if (iP==1){ // phi+
        minmax_plot[0] = 9.435;
        minmax_plot[1] = 9.475;
      }
      else{ // pt
        minmax_plot[0] = 9.4;
        minmax_plot[1] = 9.65;
      }
    }

  }
  else if (type=="sigma"){

    if (resonance=="Z"){
      if (iP==7){ // PhiCS
        minmax_plot[0] = 1.1;
        minmax_plot[1] = 1.7;
      }
      else if (iP==6){ // cosThetaCS
        minmax_plot[0] = 1.0;
        minmax_plot[1] = 1.65;
      }
      else if (iP==5){ // deta
        minmax_plot[0] = 0.8;
        minmax_plot[1] = 2.3;
      }
      else if (iP==4){ // eta-
        minmax_plot[0] = 0.8;
        minmax_plot[1] = 3.2;
      }
      else if (iP==3){ // eta+
        minmax_plot[0] = 0.8;
        minmax_plot[1] = 3.2;
      }
      else if (iP==2){ // phi-
        minmax_plot[0] = 1.0;
        minmax_plot[1] = 1.7;
      }
      else if (iP==1){ // phi+
        minmax_plot[0] = 1.0;
        minmax_plot[1] = 1.7;
      }
      else{ // pt
        minmax_plot[0] = 0;
        minmax_plot[1] = 3.0;
      }
    }
    else if (resonance=="Y1S"){
      if (iP==7){ // PhiCS
        minmax_plot[0] = 0.05;
        minmax_plot[1] = 0.16;
      }
      else if (iP==6){ // cosThetaCS
        minmax_plot[0] = 0.06;
        minmax_plot[1] = 0.12;
      }
      else if (iP==5){ // deta
        minmax_plot[0] = 0.04;
        minmax_plot[1] = 0.16;
      }
      else if (iP==4){ // eta-
        minmax_plot[0] = 0.05;
        minmax_plot[1] = 0.22;
      }
      else if (iP==3){ // eta+
        minmax_plot[0] = 0.05;
        minmax_plot[1] = 0.22;
      }
      else if (iP==2){ // phi-
        minmax_plot[0] = 0.06;
        minmax_plot[1] = 0.11;
      }
      else if (iP==1){ // phi+
        minmax_plot[0] = 0.06;
        minmax_plot[1] = 0.11;
      }
      else{ // pt
        minmax_plot[0] = 0.0;
        minmax_plot[1] = 0.26;
      }
    }

  } // End custom ranges

}

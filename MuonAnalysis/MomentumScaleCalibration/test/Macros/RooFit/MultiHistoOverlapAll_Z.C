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
#include "Alignment/OfflineValidation/plugins/TkAlStyle.cc"


using namespace ROOT::Math;

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
void MultiHistoOverlapAll_Z(string files, string labels, string colors = "", string linestyles = "", TString directory = ".", bool switchONfit = false){
  gSystem->mkdir(directory, true);
  gROOT->Reset();
  if (TkAlStyle::status() == NO_STATUS)
    TkAlStyle::set(INTERNAL);
  gROOT->ForceStyle();

  vector<string> strValidation_file;
  vector<string> strValidation_label;
  vector<string> strValidation_color;
  vector<string> strValidation_linestyle;
  splitOptionRecursive(files, strValidation_file, ',');
  splitOptionRecursive(labels, strValidation_label, ',');
  splitOptionRecursive(colors, strValidation_color, ',');
  splitOptionRecursive(linestyles, strValidation_linestyle, ',');
  int nfiles = strValidation_file.size();
  int nlabels = strValidation_label.size();
  int ncolors = strValidation_color.size();
  int nlinestyles = strValidation_linestyle.size();
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

  TH1D** histo[7];
  TF1** hfit[7];
  TFile** file = new TFile*[nfiles];
  for (int c=0; c<7; c++){
    histo[c] = new TH1D*[nfiles];
    hfit[c] = new TF1*[nfiles];
  }

  for (int f=0; f<nfiles; f++) file[f] = TFile::Open((strValidation_file[f]).c_str(), "read");

  double minmax_plot[7][2]={ { 0 } };
  int pIndex;
  TCanvas* c[7];
  for (int i=0; i<7; i++){
    TString cname = Form("c%i", i);
    c[i] = new TCanvas(cname, cname, 8, 30, 800, 800);
  }

  TLegend *leg = TkAlStyle::legend("topleft", nfiles);

  //----------------- CANVAS C0 --------------//
  pIndex=0;
  c[pIndex]->cd();

  TString histoName[7] ={
    "MassVsPhiPlus/allHistos/meanHisto",
    "MassVsPhiMinus/allHistos/meanHisto",
    "MassVsEtaPlus/allHistos/meanHisto",
    "MassVsEtaMinus/allHistos/meanHisto",
    "MassVsEtaPlusMinusDiff/allHistos/meanHisto",
    "MassVsCosThetaCS/allHistos/meanHisto",
    "MassVsPhiCS/allHistos/meanHisto"
  };
  TString xtitle[7] ={
    "#phi_{#mu+}",
    "#phi_{#mu-}",
    "#eta_{#mu+}",
    "#eta_{#mu-}",
    "#eta_{#mu+} - #eta_{#mu-}",
    "cos #theta_{CS}",
    "#phi_{CS}"
  };
  TString plotname[7] ={
    "MassVsPhiPlus_ALL",
    "MassVsPhiMinus_ALL",
    "MassVsEtaPlus_ALL",
    "MassVsEtaMinus_ALL",
    "MassVsDeltaEta_ALL",
    "MassVsCosThetaCS_ALL",
    "MassVsPhiCS_ALL"
  };
  TString fitFormula[7]={
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*x",
    "[0]+[1]*x",
    "[0]+[1]*x",
    "[0]+[1]*cos(x+[2])",
    "[0]+[1]*cos(x+[2])"
  };
  double plot_xmax[7]={
    TMath::Pi(),
    TMath::Pi(),
    2.4,
    2.4,
    4.8,
    1,
    TMath::Pi()
  };

  // Magic numbers
  int nfileslimit = 4;
  double rangeFactor[2]={ 1.01, 1.05 };
  double dampingFactor = 0;
  double deviationThreshold = 1.04;
  if (nfiles>nfileslimit) dampingFactor = 0.07/nfileslimit*(nfiles-nfileslimit);

  for (int iP=0; iP<7; iP++){
    double absMin = 9e9;
    double absMax = -9e9;
    double rangeMaxReduction = 0.02;
    if (nfiles>nfileslimit) rangeMaxReduction = rangeMaxReduction*nfileslimit/nfiles;
    double dampingFactorEff = dampingFactor;

    double avgM = 0;
    double sigmaM = 0;

    for (int f=0; f<nfiles; f++){
      histo[iP][f]=(TH1D*)file[f]->Get(histoName[iP]);

      histo[iP][f]->SetTitle("");

      histo[iP][f]->GetYaxis()->SetTitle("M_{#mu#mu} (GeV)");
      histo[iP][f]->SetLineWidth(1);
      histo[iP][f]->SetMarkerSize(1.2);

      if (strValidation_label.at(f).find("reference")!=string::npos || strValidation_label.at(f).find("Reference")!=string::npos) histo[iP][f]->SetMarkerStyle(1);
      else histo[iP][f]->SetMarkerStyle(20);

      if (ncolors!=0){
        int color = stoi(strValidation_color[f]);
        histo[iP][f]->SetLineColor(color);
        histo[iP][f]->SetMarkerColor(color);
      }
      else if (f==0){
        histo[iP][f]->SetLineColor(kBlack);
        histo[iP][f]->SetMarkerColor(kBlack);
      }
      else if (f==(nfiles-1)){
        histo[iP][f]->SetLineColor(kViolet);
        histo[iP][f]->SetMarkerColor(kViolet);
      }
      else if (f==1){
        histo[iP][f]->SetLineColor(kBlue);
        histo[iP][f]->SetMarkerColor(kBlue);
      }
      else if (f==2){
        histo[iP][f]->SetLineColor(kRed);
        histo[iP][f]->SetMarkerColor(kRed);
      }
      else if (f==3){
        histo[iP][f]->SetLineColor(kGreen+2);
        histo[iP][f]->SetMarkerColor(kGreen+2);
      }
      else if (f==4){
        histo[iP][f]->SetLineColor(kOrange+3);
        histo[iP][f]->SetMarkerColor(kOrange+3);
      }
      else if (f==5){
        histo[iP][f]->SetLineColor(kGreen);
        histo[iP][f]->SetMarkerColor(kGreen);
      }
      else if (f==6){
        histo[iP][f]->SetLineColor(kYellow);
        histo[iP][f]->SetMarkerColor(kYellow);
      }
      else if (f==7){
        histo[iP][f]->SetLineColor(kPink+9);
        histo[iP][f]->SetMarkerColor(kPink+9);
      }
      else if (f==8){
        histo[iP][f]->SetLineColor(kCyan);
        histo[iP][f]->SetMarkerColor(kCyan);
      }
      else if (f==9){
        histo[iP][f]->SetLineColor(kGreen+3);
        histo[iP][f]->SetMarkerColor(kGreen+3);
      }

      if (nlinestyles!=0){
        int linestyle = stoi(strValidation_linestyle[f]);
        histo[iP][f]->SetLineStyle(linestyle % 100);
        if (linestyle > 100)
          histo[iP][f]->SetMarkerStyle(linestyle / 100);
      }

      if (iP==0) leg->AddEntry(histo[iP][f], (strValidation_label.at(f)).c_str(), "lp");

      for (int bin=1; bin<=histo[iP][f]->GetNbinsX(); bin++){
        double bincontent = histo[iP][f]->GetBinContent(bin);
        double binerror = histo[iP][f]->GetBinError(bin);
        if (binerror==0 && bincontent==0) continue;
        absMin = min(absMin, bincontent - binerror);
        absMax = max(absMax, bincontent + binerror);
        avgM += bincontent/pow(binerror, 2);
        sigmaM += 1./pow(binerror, 2);
      }
    }
    avgM /= sigmaM;
    sigmaM = sqrt(1./sigmaM);
    for (int f=0; f<nfiles; f++){
      for (int bin=1; bin<=histo[iP][f]->GetNbinsX(); bin++){
        double bincontent = histo[iP][f]->GetBinContent(bin);
        double binerror = histo[iP][f]->GetBinError(bin);
        if (binerror==0 && bincontent==0) continue;
        if ((bincontent + binerror)>deviationThreshold*avgM) rangeMaxReduction = 0;
      }
    }
    if (nfiles>nfileslimit && rangeMaxReduction!=0) dampingFactorEff = dampingFactorEff*0.7;

    minmax_plot[iP][0] = absMin/rangeFactor[0];
    minmax_plot[iP][1] = absMax*(rangeFactor[1]+dampingFactorEff-rangeMaxReduction);
    for (int f=0; f<2 && f<nfiles; f++) histo[iP][f]->GetYaxis()->SetRangeUser(minmax_plot[iP][0], minmax_plot[iP][1]);
  }

  for (int pIndex=0; pIndex<7; pIndex++){
    for (int f=0; f<nfiles; f++){
      if (histo[pIndex][f]==0){
        cout << "Non-existent histogram in file " << f << " with canvas " << pIndex << endl;
        continue;
      }
      c[pIndex]->cd();
      histo[pIndex][f]->GetXaxis()->SetTitle(xtitle[pIndex]);
      histo[pIndex][f]->GetXaxis()->SetRangeUser(-plot_xmax[pIndex], plot_xmax[pIndex]);
      if (f==0) histo[pIndex][f]->Draw();
      else histo[pIndex][f]->Draw("same");

      hfit[pIndex][f] = new TF1(Form("fit_%i_%i", pIndex, f), fitFormula[pIndex], -plot_xmax[pIndex], plot_xmax[pIndex]);
      hfit[pIndex][f]->SetParameter(0, 90.5);
      hfit[pIndex][f]->SetParameter(1, 0);
      if (fitFormula[pIndex].Contains("[2]")) hfit[pIndex][f]->SetParameter(2, 0);
      hfit[pIndex][f]->SetLineColor(1);
      if (switchONfit){
        histo[pIndex][f]->Fit(hfit[pIndex][f], "R", "same", -plot_xmax[pIndex], plot_xmax[pIndex]);
        hfit[pIndex][f]->Draw("same");
      }
    }
    leg->Draw("same");
    TkAlStyle::drawStandardTitle();
    c[pIndex]->RedrawAxis();
    c[pIndex]->Modified();
    c[pIndex]->Update();
    c[pIndex]->SaveAs(Form("%s/%s%s", directory.Data(), plotname[pIndex].Data(), ".png"));
    c[pIndex]->SaveAs(Form("%s/%s%s", directory.Data(), plotname[pIndex].Data(), ".pdf"));

    for (int f=0; f<nfiles; f++) delete hfit[pIndex][f];
    c[pIndex]->Close();
  }

  delete leg;
  for(int f=nfiles-1;f>=0;f--) file[f]->Close();

  delete[] file;
  for (int c=0; c<7; c++){
    delete[] hfit[c];
    delete[] histo[c];
  }
}

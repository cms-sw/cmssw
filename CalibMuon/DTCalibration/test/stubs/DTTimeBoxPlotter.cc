/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/01/22 19:00:30 $
 *  $Revision: 1.5 $
 *  \author G. Cerminara - INFN Torino
 */

#include "DTTimeBoxPlotter.h"
#include "DTTimeBoxFitter.h"



#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>

#include "TFile.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TCollection.h"
#include "TSystem.h"
#include "TH2F.h"
// #include "TProfile.h"

using namespace std;

// Constructor
DTTimeBoxPlotter::DTTimeBoxPlotter( TFile *file) : theFile(file), theVerbosityLevel(0) {
  theFitter = new DTTimeBoxFitter();
  theFitter->setVerbosity(1);
}


// Destructor
DTTimeBoxPlotter::~DTTimeBoxPlotter(){}




TH1F* DTTimeBoxPlotter::plotTimeBox(int wheel, int station, int sector, const TString& drawOptions) {
  TString histoName = getHistoNameSuffix(wheel, station, sector) + "_hTimeBox";
  return plotHisto(histoName, drawOptions);
}

TH1F* DTTimeBoxPlotter::plotTimeBox(int wheel, int station, int sector, int sl, const TString& drawOptions) {
  TString histoName = getHistoNameSuffix(wheel, station, sector, sl) + "_hTimeBox";
  return plotHisto(histoName, drawOptions);
}

TH1F* DTTimeBoxPlotter::plotTimeBox(int wheel, int station, int sector, int sl, int layer,
				const TString& drawOptions) {
  TString histoName = getHistoNameSuffix(wheel, station, sector, sl, layer) + "_hTimeBox";
  return plotHisto(histoName, drawOptions);
}

TH1F* DTTimeBoxPlotter::plotTimeBox(int wheel, int station, int sector, int sl, int layer, int wire,
				const TString& drawOptions) {
  TString histoName = getHistoNameSuffix(wheel, station, sector, sl, layer, wire) + "_hTimeBox";
  return plotHisto(histoName, drawOptions);
}


void DTTimeBoxPlotter::printPDF() {
  TIter iter(gROOT->GetListOfCanvases());
  TCanvas *c;
  while( (c = (TCanvas *)iter()) ) {
    c->Print(0,"ps");
  }
  gSystem->Exec("ps2pdf *.ps");
}



TString DTTimeBoxPlotter::getHistoNameSuffix(int wheel, int station, int sector) {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" <<wheel << "_" << station << "_" << sector << "_SLall_Lall_Wall";
  theStream >> histoName;
  return TString(histoName.c_str());
}



TString DTTimeBoxPlotter::getHistoNameSuffix(int wheel, int station, int sector, int sl) {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" <<wheel << "_" << station << "_" << sector << "_SL" << sl;
  theStream >> histoName;
  return TString(histoName.c_str());
}



TString DTTimeBoxPlotter::getHistoNameSuffix(int wheel, int station, int sector, int sl, int layer) {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" <<wheel << "_" << station << "_" << sector << "_SL" << sl << "_L" << layer << "_Wall";
  theStream >> histoName;
  return TString(histoName.c_str());
}



TString DTTimeBoxPlotter::getHistoNameSuffix(int wheel, int station, int sector, int sl, int layer, int wire) {
  string histoName;
  stringstream theStream;
  theStream << "Ch_" <<wheel << "_" << station << "_" << sector << "_SL" << sl << "_L" << layer << "_W" << wire;
  theStream >> histoName;
  return TString(histoName.c_str());
}



TH1F* DTTimeBoxPlotter::plotHisto(const TString& histoName, const TString& drawOptions) {
  TH1F *histo = (TH1F *) theFile->Get(histoName.Data());
  if(histo == 0) {
    cout << "***Error: Histogram: " << histoName << " doesn't exist!" << endl;
    return 0;
  }
  static int color;
  
  TCanvas *c;
  if(!drawOptions.Contains("same")) {
    color = 1;
    c = newCanvas("c_"+histoName);
    c->cd();
    histo->SetLineColor(color);
  } else {
    color ++;
    histo->SetLineColor(color);
  }
  histo->Draw(TString("h"+drawOptions).Data());

  if(drawOptions.Contains("fit")) {
    theFitter->fitTimeBox(histo);
  }


  return histo;
}




TH2F* DTTimeBoxPlotter::plotHisto2D(const TString& histoName, const TString& drawOptions) {
  TH2F *histo = (TH2F *) theFile->Get(histoName.Data());
  if(histo == 0) {
    cout << "***Error: Histogram: " << histoName << " doesn't exist!" << endl;
    return 0;
  }
  static int color;

  TCanvas *c;
  if(!drawOptions.Contains("same")) {
    color = 1;
    c = newCanvas("c_"+histoName);
    c->cd();
    histo->SetLineColor(color);
  } else {
    color ++;
    histo->SetLineColor(color);
  } 
  histo->Draw(TString("h"+drawOptions).Data());
  return histo;
}



TCanvas * DTTimeBoxPlotter::newCanvas(TString name, TString title,
				    int xdiv, int ydiv, int form, int w){
  static int i = 1;
  if (name == "") {
    name = TString("Canvas "+i);
    i++;
  }
  TCanvas *c = 0;
  if (title == "") title = name;
  if (w<0) {
    c = new TCanvas(name,title, form);
  } else {
    c = new TCanvas(name,title,form,w);
  }
  if (xdiv*ydiv!=0) c->Divide(xdiv,ydiv);
  c->cd(1);
  return c;
}

TCanvas * DTTimeBoxPlotter::newCanvas(TString name, int xdiv, int ydiv, int form, int w) {
  return newCanvas(name, name,xdiv,ydiv,form,w);
}
TCanvas * DTTimeBoxPlotter::newCanvas(int xdiv, int ydiv, int form) {
  return newCanvas("","",xdiv,ydiv,form);
}
TCanvas * DTTimeBoxPlotter::newCanvas(int form)
{
  return newCanvas(0,0,form);
}

TCanvas * DTTimeBoxPlotter::newCanvas(TString name, int form, int w)
{
  return newCanvas(name, name, 0,0,form,w);
}


// Set the verbosity of the output: 0 = silent, 1 = info, 2 = debug
void DTTimeBoxPlotter::setVerbosity(unsigned int lvl) {
  theVerbosityLevel = lvl;
  theFitter->setVerbosity(lvl);
}

void DTTimeBoxPlotter::setInteractiveFit(bool isInteractive) {
  theFitter->setInteractiveFit(isInteractive);
}

void DTTimeBoxPlotter::setRebinning(int rebin) {
  theFitter->setRebinning(rebin);
}


//#include "stdlib"
//#include "stdio"

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TKey.h"
#include "TObject.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGaxis.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLegend.h"

std::vector< TString > labels;
std::vector<float> values;
float SumValues[3];
std::vector<float> xvalue;
std::vector<TDatime> Dates;


void CreatePlots(TString filename,TString XTitle="", TString YTitle="", bool boolLog=true, bool cumulate=true){

  //  TString filename_=filename;
  ifstream infile;
  infile.open(filename.Data());
  if (!infile.is_open())
    return;

  cout << "filename " << filename << endl;

  char line[1024];
  int count=0;
  while (infile.good()){
    infile.getline(line,1024);
    //cout << line << endl;
    char * pch = strtok (line,"|");
    int ic=0;
    while (pch != NULL){
      if (ic==0 || ic ==3){
	ic++;
	pch = strtok (NULL, "|");
	continue;
      }
      //      cout << pch << endl;
      if (count){
	if (ic==1){
	  xvalue.push_back(atoi(pch));
	}
	else if (ic==2)
	  Dates.push_back(TString(pch).ReplaceAll(".","-").Data());
	else
	  values.push_back(atof(pch));
      }
      pch = strtok (NULL, "|");
      ic++;
    }
    count++;
  }

  labels.push_back("RawData");
  labels.push_back("RecoData");
  labels.push_back("RecoDataFNAL");

  TCanvas *c1 = new TCanvas();
  const size_t m = labels.size();
  int n = values.size()/m; //labels.size();
  TGraph* gr[m];
  TMultiGraph *mg = new TMultiGraph();
  TLegend *tleg = new TLegend(0.2,.85,.35,0.65);

  //  double* x = (double*) malloc(m*n*sizeof(double));
  for (size_t i=0;i<n;++i) {
    for (size_t j=0;j<m;++j) {
      if (!i){
	gr[j]= new TGraph(n);
	gr[j]->SetMarkerStyle(21+j);
	gr[j]->SetMarkerColor(j+2);
	mg->Add(gr[j],"pl");
	tleg->AddEntry(gr[j],labels[j],"p");
      }
      
      if (cumulate==false){
	if (values[i*m+j]==0)
	  gr[j]->SetPoint(i,xvalue[i],.1);
	else
	  gr[j]->SetPoint(i,xvalue[i],values[i*m+j]);
      } else {
	SumValues[j]+=values[i*m+j];
	gr[j]->SetPoint(i,xvalue[i],SumValues[j]);
      }
      //    cout << xvalue[i]<< " " << values[i*m+j] << endl;
    }
  }

  if (boolLog)
    c1->SetLogy();

  c1->SetGridy();
  c1->SetGridx();
  mg->Draw("a");
  mg->GetXaxis()->SetTitle(XTitle);
  mg->GetYaxis()->SetTitle(YTitle);
  tleg->Draw();

  float ymax= boolLog? pow(10,gPad->GetUymax()):gPad->GetUymax();

  int X0=Dates[0].Convert();
  gStyle->SetTimeOffset(X0);
  TGaxis *axis = new TGaxis(gPad->GetUxmin(),ymax,
			    gPad->GetUxmax(), ymax,Dates[0].Convert()-X0,Dates[n-1].Convert()-X0,508,"-t");

  axis->SetTimeOffset(X0,"gmt");
  axis->SetTimeFormat("%m/%d");
  axis->SetTitle("Date");
  //axis->SetLineColor(kRed);
  //axis->SetLabelColor(kRed);

  axis->Draw();
  
  gPad->Update();
  TString logFlag="";
  if (boolLog)
    logFlag="_logY";
  if (cumulate)
    c1->Print(filename.ReplaceAll(".txt",logFlag+"_integrated.gif"));
  else
    c1->Print(filename.ReplaceAll(".txt",logFlag+"_diff.gif"));

  delete axis;
};

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
std::vector<unsigned long> Dates;
float xmin,xmax,ymax;

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
	  Dates.push_back(TDatime(TString(pch).ReplaceAll(".","-").Data()).Convert());
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
	gr[j]->SetPoint(i,xvalue[i],SumValues[j]/1e06);
      }
    }
  }


 gStyle->SetNdivisions(510);

  c1->SetGridy();
  c1->SetGridx();
  if (boolLog)
    c1->SetLogy();


  mg->Draw("a");
  gPad->Update();
  
  if (cumulate==false)
      mg->GetYaxis()->SetTitle("NEntries");
  else
    mg->GetYaxis()->SetTitle("NEntries (10^{6})");

  TAxis* xaxis=mg->GetXaxis();

  xaxis->SetTitle("Date");
  xaxis->SetTitleOffset(2.0);

  int lastDay=0;
  int lastMonth=0;
  int lastBin=0;
  int lastTime=0;
  int month, day, time;
  char label[64];

  for (size_t i=0;i<xaxis->GetNbins()+1;++i){
    //   cout << i << " " << TString(xaxis->GetBinLabel(i)) << " " <<  xaxis->GetBinLowEdge(i)<< " " <<  xaxis->GetBinUpEdge(i)<< endl;
    int upper=(int) xaxis->GetBinUpEdge(i);
    int lower=(int) xaxis->GetBinLowEdge(i);
    std::vector<float>::const_iterator p=upper_bound(xvalue.begin(),xvalue.end(),lower);
    std::vector<float>::const_iterator q=upper_bound(xvalue.begin(),xvalue.end(),upper);
    if (q!=xvalue.begin())
    q--;
    //cout <<  i << " " << " " <<  lower<< " " <<  upper<< " " << *p << " " << *q << endl;
    if (p!=xvalue.end() && q!=xvalue.end()
	&&
	*p<upper && *q>lower 
	&&
	TDatime(Dates[q-xvalue.begin()]).GetDate() - TDatime(Dates[p-xvalue.begin()]).GetDate() < 86400
	){
      day=TDatime(Dates[p-xvalue.begin()]).GetDay();
      month=TDatime(Dates[p-xvalue.begin()]).GetMonth();
      time=Dates[p-xvalue.begin()];
      if ( i-lastBin>4  //reduce the number of labels
	   && 
	   time-lastTime > 5*86400 //ask for dates after 5 days
	   // && (month!=lastMonth || day!=lastDay)
	   ){
	//cout << "-------- " << *p << " " << TDatime(Dates[p-xvalue.begin()]).AsString() << endl;
	sprintf(label,"%02d/%02d",day,month);
	xaxis->SetBinLabel(i,label);
	//lastMonth=month;
	//lastDay=day;
	lastTime=time;
	lastBin=i;
      }
    }
  }
  tleg->Draw();
  
  gPad->Update();

  xmin=mg->GetXaxis()->GetXmin();
  xmax=mg->GetXaxis()->GetXmax();
  ymax=mg->GetYaxis()->GetXmax();

  //cout << "min max " << xmin << " " << xmax << " " << ymax << endl;
  //cout << Dates[0] << " " << Dates[n-1] << endl;

  TGaxis *axis = new TGaxis(xmin,ymax,xmax,ymax,xmin,xmax,510,"-");
  axis->SetTitle("RunNb");
  axis->Draw();
  

  gPad->Update();
   TString logFlag="";
   if (boolLog)
     logFlag="_logY";
   if (cumulate)
     c1->Print(filename.ReplaceAll(".txt",logFlag+"_integrated.gif"));
   else
     c1->Print(filename.ReplaceAll(".txt",logFlag+"_diff.gif"));
};

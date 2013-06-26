#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"


void drawall(char* f1, char* f2 ,string pathName,char* label1, char* label2, char* var , string outname){
 
  gStyle->SetOptTitle(1);
  
  TFile* file1 = TFile::Open(f1);
  TFile* file2 = TFile::Open(f2);

  TH1F* hf1=0, *hf2=0;

  TH1F* num1=0, *denom1=0;

  char* name1=0, *name2=0;

  const int npaths = 1;
  string paths[npaths]={"DQMData/ElectronOf"+pathName+"DQM"};
  string Histo_paths[npaths];
  for(int i=0;i<npaths;i++){Histo_paths[i]=paths[i]+"/total eff";cout<<Histo_paths[i]<<endl;}

  string fname= outname+"(";
  TCanvas* c1= new TCanvas("c1","c1");
  c1->SetFillColor(0);
  c1->Divide(2,2);

  string out_file=pathName+"Plots.root";
  TFile* f=new TFile(out_file.c_str(),"RECREATE");
  
  for(int path=0; path< npaths ; path++){
    
    hf1 = (TH1F*)file1->Get(Histo_paths[path].c_str());
    hf2 = (TH1F*)file2->Get(Histo_paths[path].c_str());
    
    int plotnr = 0;
    TGraphAsymmErrors* graph1;TGraphAsymmErrors* graph2;
    
    for(int filter=1; filter < hf1->GetNbinsX() -2 ; filter++){
      name1 = hf1->GetXaxis()->GetBinLabel(filter);
      name2 = hf1->GetXaxis()->GetBinLabel(filter+1);
      
      string numname = paths[path] + "/"+ name2 + var;
      string denomname = paths[path] + "/"+ name1 + var;
      string title = paths[path]+name2+var+"Efficiency";
      
      num1   =  new TH1F( *(TH1F*)file1->Get(numname.c_str()) );
      denom1 =  new TH1F( *(TH1F*)file1->Get(denomname.c_str()) );
      num2   =  new TH1F( *(TH1F*)file2->Get(numname.c_str()) ); 
      denom2 =  new TH1F( *(TH1F*)file2->Get(denomname.c_str() ));
      
      graph1=new TGraphAsymmErrors;
      graph1->SetName((title+"1").c_str());
      graph1->BayesDivide(num1,denom1);
      graph1->SetLineColor(2);
      graph1->GetXaxis()->SetTitle(var);
      graph1->GetYaxis()->SetTitle("Efficiency");
      
      graph2=new TGraphAsymmErrors;
      graph2->SetName((title+"2").c_str());
      graph2->BayesDivide(num2,denom2);
      graph2->SetLineColor(4);
      graph2->SetMarkerStyle(24);
      graph2->GetXaxis()->SetTitle(var);
      graph2->GetYaxis()->SetTitle("Efficiency");

      TLegend *legend = new TLegend(0.4,0.2,0.55,0.4,"");
      l1 = legend->AddEntry(graph1,label1,"lp");
      l1 = legend->AddEntry(graph2,label2,"lp");
      
      TPaveText* pave=new TPaveText();
      pave->AddText(name2);
      
      if(plotnr%4==0){
	c1->Clear("D");
      }
      c1->cd(plotnr%4 + 1);

      graph1->Draw("AP");
      graph2->Draw("Psame");
      legend->Draw();      
      pave->Draw();

      plotnr++;
   
      if(plotnr%4==0){
	cout<<fname<<endl;
	c1->Print(fname.c_str());
      }   
      std::cout <<"::"<<filter<<"::"<<numname << "::"<<num1->GetEntries()<<std::endl;  
      std::cout <<"::"<<filter<<"::"<<denomname << "::"<<denom1->GetEntries()<<std::endl;  
      graph1->Write();graph2->Write();
    }
    
    if(plotnr%4!=0){
      c1->Print(fname.c_str());
    }
  }
  c1->Clear();
  fname=outname+")";
  c1->Print(fname.c_str());


}


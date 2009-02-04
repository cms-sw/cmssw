#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"


void drawall(char* f1, char* f2 ,char* label1, char* label2, char* var , string outname){
  TFile* file1 = TFile::Open(f1);
  TFile* file2 = TFile::Open(f2);

  TH1F* hf1=0, *hf2=0;

  TH1F* num1=0, *denom1=0;

  char* name1=0, *name2=0;

  const int npaths = 3;
  string paths[npaths]={"singleElectron","singleElectronRelaxed","singleElectronLargeWindow"};
  string Histo_paths[npaths];
  for(int i=0;i<npaths;i++){Histo_paths[i]=paths[i]+"DQM/total eff";}

  string fname= outname + "(";
  TCanvas* c1= new TCanvas("c1","c1");
  c1->SetFillColor(0);
  c1->Divide(2,2);

  for(int path=0; path< npaths ; path++){
    
    hf1 = (TH1F*)file1->Get(Histo_paths[path].c_str());
    hf2 = (TH1F*)file2->Get(Histo_paths[path].c_str());
    
    int plotnr = 0;

    for(int filter=1; filter < hf1->GetNbinsX() -2 ; filter++){
      name1 = hf1->GetXaxis()->GetBinLabel(filter);
      name2 = hf1->GetXaxis()->GetBinLabel(filter+1);
      std::cout << name1 << std::endl;   
      
      string numname = paths[path] + "DQM/"+ name2 + var;
      string denomname = paths[path] + "DQM/"+ name1 + var;
      
      std::cout << denomname << std::endl;  

      num1   =  new TH1F( *(TH1F*)file1->Get(numname.c_str()) );
      denom1 =  new TH1F( *(TH1F*)file1->Get(denomname.c_str()) );
      num2   =  new TH1F( *(TH1F*)file2->Get(numname.c_str()) );
      std::cout << name1 << std::endl;  

      //*********************************************************
      // DANGER: terrible clutch for differing filter names in 200 and 183
      if(!strcmp(name2,"hltL1seedSingle")){
	denomname = paths[path] + "DQM/"+ "hltL1seedSingleEgamma" + var;
      }
      if(!strcmp(name2,"hltL1seedRelaxedSingle")){
	denomname = paths[path] + "DQM/"+ "hltL1seedRelaxedSingleEgamma" + var;
      }
      if(!strcmp(name2,"hltL1seedDouble")){
	denomname = paths[path] + "DQM/"+ "hltL1seedDoubleEgamma" + var;
      }
      if(!strcmp(name2,"hltL1seedRelaxedDouble")){
	denomname = paths[path] + "DQM/"+ "hltL1seedRelaxedDoubleEgamma" + var;
      }
      // ********************************************************
      denom2 =  new TH1F( *(TH1F*)file2->Get(denomname.c_str() ));
      
      num1->Sumw2();
      denom1->Sumw2();
      num2->Sumw2();
      denom2->Sumw2();

      num1->Divide(num1,denom1,1.,1.,"b");
      num2->Divide(num2,denom2,1.,1.,"b");
   
      string title = paths[path] + " : " + name2 + ";" + var + " ;Efficiency";
      num1->SetTitle(title.c_str());
      num1->SetLineColor(2);
      num2->SetLineColor(4);
 
      TLegend *legend = new TLegend(0.4,0.2,0.55,0.4,"");
      TLegendEntry* l1 = legend->AddEntry(num1,label1,"l");
      l1->SetTextSize(0.1);
      l1 = legend->AddEntry(num2,label2,"l");
      l1->SetTextSize(0.1);


      //std::cout << plotnr%4 << std::endl;
      if(plotnr%4==0){
	c1->Clear("D");
      }
      c1->cd(plotnr%4 + 1);

      num1->Draw();
      num2->Draw("same");
      legend->Draw();      

      plotnr++;
   
      if(plotnr%4==0){
	c1->Print(fname.c_str());
      }   
    }
    
    if(plotnr%4!=0){
      c1->Print(fname.c_str());
    }
  }
  c1->Clear();
  fname=outname+")";
  c1->Print(fname.c_str());


}



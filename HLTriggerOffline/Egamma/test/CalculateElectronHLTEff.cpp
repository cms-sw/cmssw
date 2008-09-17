//gcc `root-config --cflags --libs` CalculateElectronHLTEff.cpp -o CalculateElectronHLTEff.exe

//gcc -pthread -m32 -I/afs/cern.ch/cms/sw/slc4_ia32_gcc345/lcg/root/5.14.00g-CMS18b//include -L/afs/cern.ch/cms/sw/slc4_ia32_gcc345/lcg/root/5.14.00g-CMS18b//lib -lCore -lCint -lHist -lGraf -lGraf3d -lGpad -lTree -lRint -lPostscript -lMatrix -lPhysics -pthread -lm -ldl -rdynamic CalculateElectronHLTEff.cpp -o CalculateElectronHLTEff.exe

#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>

#include "TFile.h"
#include "TH1F.h"

using namespace std;
int main(int argc, char* argv[])
{
  if(argc < 3 ){
    cout<<"Usage: ./CalculateEgammaHLTEff.exe histofile1.root histofile2.root"<<endl;
    return -1;
  }
  //cout<<"Ciao"<<endl;
  TFile file1(argv[1]);
  TFile file2(argv[2]);
  ofstream outfile ("test.tex");
  outfile<<"\\documentclass{article}"<<endl;
  outfile<<"\\usepackage{times}"<<endl;
  outfile<<"\\usepackage{color}"<<endl;
  outfile<<"\\usepackage{amssymb}"<<endl;
  outfile<<"\\begin{document}"<<endl;
  const int npaths = 3;
  string paths[npaths]={"singleElectron","singleElectronRelaxed","singleElectronLargeWindow"};
  string Histo_paths[npaths];
  for(int i=0;i<npaths;i++){Histo_paths[i]=paths[i]+"DQM/total eff";}
  //Single electron path
  TH1F* h1=0, *h2=0;
  TH1F* hf1=0, *hf2=0;

  for(int path=0; path<npaths; path++){
   
    cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<endl;
    cout<<"@@@@@@@@@@@@              "<< paths[path]<<"             \t    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<endl;
    cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<endl;
    hf1 = (TH1F*)file1.Get(Histo_paths[path].c_str());
    hf2 = (TH1F*)file2.Get(Histo_paths[path].c_str());
    if(hf1 != 0 && hf2 !=0 ){
      int Maxbin = hf1->GetNbinsX();
      if (hf2->GetNbinsX() != Maxbin){
	cout<<"AAAA histos for path "<< paths[path] << " have different number of bins"<<endl;
	continue;
      }
      string nm1=paths[path]+"_1";
      string nm2=paths[path]+"_2";
      h1= new TH1F(nm1.c_str(),nm1.c_str(),Maxbin,0,Maxbin);
      h2= new TH1F(nm2.c_str(),nm2.c_str(),Maxbin,0,Maxbin);
      
      for(int bin=0;bin < Maxbin-1; bin++){
	h1->SetBinContent(bin+2,hf1->GetBinContent(bin));
	h1->GetXaxis()->SetBinLabel(bin+2,hf1->GetXaxis()->GetBinLabel(bin));
	h2->SetBinContent(bin+2,hf2->GetBinContent(bin));
	h2->GetXaxis()->SetBinLabel(bin+2,hf2->GetXaxis()->GetBinLabel(bin));
	
      }
      h1->SetBinContent(1,hf1->GetBinContent(Maxbin-1));
      h1->SetBinContent(2,hf1->GetBinContent(Maxbin));
      h1->GetXaxis()->SetBinLabel(1,"Total");
      h1->GetXaxis()->SetBinLabel(2,"Gen");

      h2->SetBinContent(1,hf2->GetBinContent(Maxbin-1));
      h2->SetBinContent(2,hf2->GetBinContent(Maxbin));
      h2->GetXaxis()->SetBinLabel(1,"Total");
      h2->GetXaxis()->SetBinLabel(2,"Gen");

      outfile<<"\\begin{tabular}{ | c || c | c | }"<<endl;
      outfile<<"\\hline"<<endl;
      outfile<<"\\multicolumn{3}{|c|}{"<<paths[path]<<"}\\\\"<<endl;
      outfile<<"\\hline"<<endl;
      outfile<<" module & 2\\_1\\_0\\_pre6 & 2\\_1\\_4 \\\\"<<endl;
      outfile<<"\\hline"<<endl;
      outfile<<" Total Events & "<< h1->GetBinContent(1) <<" & " <<h2->GetBinContent(1)<<" \\\\"<<endl;
      outfile<<"\\hline"<<endl;
      
      cout<<"Total events: "<<h1->GetBinContent(1)<<"  and  "<<h2->GetBinContent(1)<<endl;
      for(int bin=2; bin < Maxbin+1; bin++){
	//name=h->GetXaxis()->GetBinLabel(bin);
	string lab1(h1->GetXaxis()->GetBinLabel(bin));
	string lab2(h2->GetXaxis()->GetBinLabel(bin));
	if(lab1 != lab2 ){
	  cout<<"AAAA histos for path "<< paths[path] << " have different labes for bin "<<bin<<" : "<<lab1<<" || "<<lab2<<endl;
	  //  break;
	}
	float den1 = h1->GetBinContent(bin-1);
	float den2 = h2->GetBinContent(bin-1);
	if(den1 >0 && den2 >0 ){
	  float eff1=h1->GetBinContent(bin)/den1;
	  float sigma1 = sqrt(eff1*(1-eff1)/den1);

	  float eff2=h2->GetBinContent(bin)/den2;
	  float sigma2 = sqrt(eff2*(1-eff2)/den2);
	  
	  float thr = 0.015;//could be tuned on sigma1,2
	  if(fabs(eff1-eff2)<thr){
	    cout<<lab1<<" Eff1: "<<fixed<<setprecision(2)<<eff1*100<<" +/- "<<sigma1*100<<" %"<<" ; Eff2: "<<eff2*100<<" +/- "<<sigma2*100<<" %"<<endl;	
	    outfile<<lab1<< " & "<<fixed<<setprecision(2)<<  eff1*100<<" $\\pm$ "<< sigma1*100 <<" & "<<eff2*100 <<" $\\pm$ "<< sigma2*100<<" \\\\"<<endl;
	  }
	  else{
	    cout<<"AAAABBBB "<<lab1<<" Eff1: "<<fixed<<setprecision(2)<<eff1*100<<" +/- "<<sigma1*100<<" %"<<" ; Eff2: "<<eff2*100<<" +/- "<<sigma2*100<<" %"<<endl;	
	    outfile<<"\\textcolor{red}{\\textbf{"<<lab1<< "}} & \\textcolor{red}{\\textbf{"<<fixed<<setprecision(2)<<  eff1*100<<" $\\pm$ "<< sigma1*100 <<"}} & \\textcolor{red}{\\textbf{"<<eff2*100 <<" $\\pm$ "<< sigma2*100<<"}} \\\\"<<endl;
	  }
	}
	else{cout<<"AAAA no event found in: "<< h1->GetXaxis()->GetBinLabel(bin-1)<<endl;}
      

      }
    
      outfile<<"\\hline"<<endl;
      float L1acc1=h1->GetBinContent(3);
      float L1acc2=h2->GetBinContent(3);
      if(L1acc1 >0 && L1acc2>0 ){

	float eff1=h1->GetBinContent(Maxbin)/L1acc1;
	float sigma1 = sqrt(eff1*(1-eff1)/L1acc1);

	float eff2=h2->GetBinContent(Maxbin)/L1acc2;
	float sigma2 = sqrt(eff2*(1-eff2)/L1acc2);
	 float thr = 0.015;//could be tuned on sigma1,2
	  if(fabs(eff1-eff2)<thr){
	    cout<<"HLT/L1: "<<fixed<<setprecision(2)<< eff1*100 <<" +/- "<<sigma1*100<<" %"<< " |||| " << eff2*100 <<" +/- "<<sigma2*100<<" %"<<endl;
	    outfile<<"HLT/L1  & "<<fixed<<setprecision(2)<<  eff1*100<<" $\\pm$ "<< sigma1*100 <<" & "<<eff2*100 <<" $\\pm$ "<< sigma2*100<<" \\\\"<<endl;
	  }
	  else{
	    cout<<"AAAABBBB HLT/L1: "<<fixed<<setprecision(2)<< eff1*100 <<" +/- "<<sigma1*100<<" %"<< " |||| " << eff2*100 <<" +/- "<<sigma2*100<<" %"<<endl;
	     outfile<<"\\textcolor{red}{\\textbf{HLT/L1}} & \\textcolor{red}{\\textbf{"<<fixed<<setprecision(2)<<  eff1*100<<" $\\pm$ "<< sigma1*100 <<"}} & \\textcolor{red}{\\textbf{"<<eff2*100 <<" $\\pm$ "<< sigma2*100<<"}} \\\\"<<endl;
	  }
      }
      else{
	cout<<"AAAA No L1 accepted"<<endl;
      }
      outfile<<"\\hline"<<endl;
      float Genacc1=h1->GetBinContent(2);
      float Genacc2=h2->GetBinContent(2);
      if(Genacc1 >0 && Genacc2>0 ){

	float eff1=h1->GetBinContent(Maxbin)/Genacc1;
	float sigma1 = sqrt(eff1*(1-eff1)/Genacc1);

	float eff2=h2->GetBinContent(Maxbin)/Genacc2;
	float sigma2 = sqrt(eff2*(1-eff2)/Genacc2);
	 float thr = 0.015;//could be tuned on sigma1,2
	  if(fabs(eff1-eff2)<thr){
	    cout<<"HLT/Gen: "<<fixed<<setprecision(2)<< eff1*100 <<" +/- "<<sigma1*100<<" %"<< " |||| " << eff2*100 <<" +/- "<<sigma2*100<<" %"<<endl;
	    outfile<<"HLT/Gen  & "<<fixed<<setprecision(2)<<  eff1*100<<" $\\pm$ "<< sigma1*100 <<" & "<<eff2*100 <<" $\\pm$ "<< sigma2*100<<" \\\\"<<endl;
	  }
	  else{
	    cout<<"AAAABBBB HLT/Gen: "<<fixed<<setprecision(2)<< eff1*100 <<" +/- "<<sigma1*100<<" %"<< " |||| " << eff2*100 <<" +/- "<<sigma2*100<<" %"<<endl;
	     outfile<<"\\textcolor{red}{\\textbf{HLT/Gen}} & \\textcolor{red}{\\textbf{"<<fixed<<setprecision(2)<<  eff1*100<<" $\\pm$ "<< sigma1*100 <<"}} & \\textcolor{red}{\\textbf{"<<eff2*100 <<" $\\pm$ "<< sigma2*100<<"}} \\\\"<<endl;
	  }
      }
      else{
	cout<<"AAAA No Gen accepted"<<endl;
      }
      outfile<<"\\hline"<<endl;
      outfile<<"\\end{tabular}"<<endl;
      outfile<<"\\newline"<<endl;
      outfile<<"\\vspace{ 0.8 cm}"<<endl;
       outfile<<"\\newline"<<endl;
      cout<<"########################################################################################"<<endl<<endl;
    }  
  }
  outfile<<"\\end{document}"<<endl;
  system("latex test.tex");
  system("dvips test.dvi -o a.ps");
  return 1;
}

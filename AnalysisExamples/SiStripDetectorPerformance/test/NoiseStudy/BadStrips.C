#include <iostream>
#include <stdio.h>
#include <string.h>
#include <sstream>

//Global variables

float LowTh; //sigmas
float HighTh; //sigmas
float AbsLowThNoise; //noise
float AbsHighThNoise; //noise

Float_t Sbin,S2bin,Mean,Mean2,stddev,diff;

char filename[128];

TFile *f, *g;
TPostScript *ps;
TCanvas *C;



///comparing new and old badstrips

void compare(TH1F *h1, TH1F *h2, TH1F *h){
  int difbad=0;


  for(int i=1; i<=h1->GetNbinsX(); i++){ 	 
    if ((h1->GetBinContent(i))!=(h2->GetBinContent(i))){
      //cout<<" badstrip number "<< i << " of " << h2->GetName() <<" is different!!!! "<< endl;
      difbad++;
    }	
   }
  if (difbad!=0){
    cout<<" the number of different badstrips in module " << h1->GetTitle() << " is "<< difbad <<"\n"<< endl;
    char title[128];
    sprintf(title,"Bad Strips for mod %s",h1->GetTitle());
    THStack* stack =new THStack(title,title);
    h1->SetFillColor(kBlue);
    h2->SetFillColor(kRed);
    stack->Add(h1);
    stack->Add(h2);
    C->cd(1);
    stack->Draw();
    C->cd(2);
    h->Draw();
    C->Draw();
    C->Update();
    ps->NewPage();   
  }else{
    cout<< "every badstrip is the same"<< endl;
  }
}
  


void apvStudy(int iApv, TH1F *histo, TH1F *histo1){
  Int_t Nbin=0;

  int Nbads=0;
  int ibinStart= iApv*128; //iApv=0 ibinStart=0; iApv=1 ibinStart=128; etc
  int ibinStop= (iApv+1)*128; //iApv=0 ibinStop=128; iApv=1 ibinStop=256; etc
  for (Int_t i=ibinStart; i<ibinStop; i++){
    Sbin += histo->GetBinContent(i+1);
    S2bin += histo->GetBinContent(i+1)*histo->GetBinContent(i+1);
    Nbin++;    
  } 
  
  Mean = Sbin/Nbin;
  Mean2 = S2bin/Nbin;
  stddev=sqrt(Mean2-Mean*Mean);

  for (Int_t i=ibinStart; i<ibinStop; i++)
    {
      if(histo->GetBinContent(i+1)>AbsLowThNoise && histo->GetBinContent(i+1)<AbsHighThNoise){
	if(histo->GetBinContent(i+1)>Mean+HighTh*stddev || histo->GetBinContent(i+1)<Mean-LowTh*stddev){
	  std::cout<< "!!!!!!  Module " << histo->GetTitle() << " Strip number  " << i << "  is a badstrip !!!!!"<< std::endl;
	  histo1->SetBinContent(i+1,1.);
	  Nbads++;
	}
      }
      else{
	std::cout<< "******  Module " << histo->GetTitle() << " Strip Noise= "<<histo->GetBinContent(i+1) <<" OUT RANGE:("<< AbsLowThNoise<<" , "<<AbsHighThNoise<< ")  Strip number   "<< i << "  is a badstrip !!!!!"<< std::endl;
	  histo1->SetBinContent(i+1,1.);
	  Nbads++;
      }
 if(Nbads > 20&&i==ibinStop-1) std::cout<< "&&&&&&  Module " << histo->GetTitle() <<"has "<<Nbads<< " bad strips "<< std::endl;
    }
 


  if(Nbads==0){
    //    cout<< "     All strips are good! "<<endl;
  }
  //  cout<< " ____________  "<<endl;
}


BadStrips(char *inputfilename, char  *outputfilename, float LowTh_=3, float HighTh_=5, float AbsLowThNoise_=2, float AbsHighThNoise_=6){

  LowTh=LowTh_;
  HighTh=HighTh_;
  AbsLowThNoise=AbsLowThNoise_;
  AbsHighThNoise=AbsHighThNoise_;
  //char histos;
  int NTotalBins;
  float xlow=0;
  
  f=new TFile(inputfilename); 
  g=new TFile(outputfilename,"RECREATE"); 
  g->mkdir("NewBadStrips");
  g->mkdir("Noises");

  C= new TCanvas();
  C->Divide(1,2);





  char psfilename[128];
  char * pch;
  strcat(psfilename,outputfilename);
  pch = strstr(psfilename,".root");
  strncpy (pch,".ps\0",4);
  cout << "psfilename " << psfilename << endl; 
  ps = new TPostScript(psfilename,121);


  f->cd("Noises");
  
  TIter nextkey(gDirectory->GetListOfKeys());
  TKey *key;
  while (key = (TKey*)nextkey()) {
    TH1F* h = (TH1F*)key->ReadObj();
  
    char *titolo;
    titolo = h->GetTitle();
    if (strstr(titolo,"Cumulative"))
      continue;
    char mybadTitle[200];
    sprintf(mybadTitle,"BadStrips_%s",&titolo[7]);
   

    TH1F *h1  = new TH1F(mybadTitle,mybadTitle,h->GetNbinsX(),-0.5,h->GetNbinsX()-0.5);


    for (int iApv=0;iApv<(Int_t)((h->GetNbinsX())/128); iApv++){
      // std::cout << "\n\nAPV i= " << iApv << "  of  "<< (Int_t)((h->GetNbinsX())/128) << " APVs in module" << &titolo[23] << "\n\n" << std::endl;
      Sbin=0;
      S2bin=0;
      Mean=0;
      Mean2=0;
      stddev=0;
      diff=0;
      apvStudy(iApv,h,h1);
    }  
    g->cd("NewBadStrips");
    h1->Write();
    g->cd("Noises");
    h->Write();

    char badstripTitle[128];
    sprintf(badstripTitle,"BadStrips/");
    strcat(badstripTitle,h1->GetTitle());

    TH1F* h2 = (TH1F*) f->Get(badstripTitle);///////h2 is in BadStrips /////h is in NewBadStrips
    
    compare(h1,h2,h);
    f->cd("Noises");
   
  }

  //save();

  f->Close();     
  g->Close();
  ps->Close();
}



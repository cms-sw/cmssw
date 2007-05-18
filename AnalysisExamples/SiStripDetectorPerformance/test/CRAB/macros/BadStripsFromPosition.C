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

char * Mask;
stringstream ss;
TFile *f;
double _prob;
static const size_t _Mask_LENGTH = 100;
static const size_t _cbads_LENGTH = 96;

TH2F* CollectivePedHisto;

struct StripStruct{
  int N;
  long double prob;
  long double mean;
};


void badStripStudy(TH1F *histo);
void CorrelatePedestals(TH1F *histo,std::map<short,StripStruct>& mBadStrip);

void Navigate(){
  TIter nextkey(gDirectory->GetListOfKeys());
  TKey *key;
  while (key = (TKey*)nextkey()) {
    TObject *obj = key->ReadObj();
    //std::cout << " object " << obj->GetName() << " " << obj->GetTitle()<< std::endl;

    if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      //cout << "Found subdirectory " << obj->GetName() << " " << obj->GetTitle()<< " " << ((TDirectory*)obj)->GetPath()<< endl;

      f->cd(obj->GetTitle());
     
      Navigate();

      f->cd("..");
    } else if ( obj->IsA()->InheritsFrom( "TH1" ) ) {

      //std::cout << "Found object " << obj->GetName() << " " << obj->GetTitle()<< std::endl;

    if (strstr(obj->GetTitle(),"cPos_Single")==NULL)
      continue;
    
    TH1F* h = (TH1F*)key->ReadObj();
    
    badStripStudy(h);
    }
  }
}

void BitAssign(int n, char* bitword){

  //The badstrip mask is an array of 768 bits, one bit for each strip
  //if a strip is bad, the corrensponding bit is rised to 1
  //the bitword is a char[96]
  //This function rise at 1 the bit corresponding to a bad strip


  size_t block=int(n/8);
  size_t bitshift=n%8;
  //cout << "BitAssign " << n << " " << block << " " << bitshift << endl;
  short uno=1;
  bitword[block]=bitword[block] & ( (~( uno << bitshift)) & 0xFF );
}
void unpack(const char* bitword){

  size_t block;
  size_t bitshift;
  for (size_t i=0;i<768;++i){
    block=int(i/8);
    bitshift=i%8;
    int num=(( bitword[4+block] & 0xFF) >> bitshift ) & 0x1;
    unsigned int detid;
    memcpy((void*)&detid,(void*)bitword,4);

    if (num==0)
      std::cout << i << " " << detid << " " << block << " " << bitshift << " unpack " << num << endl;
  }
}

void iterate(TH1F* histo,std::map<short,StripStruct>& mBadStrip){
  double diff=1.-_prob; 
  int ibinStart= 1; 
  int ibinStop= histo->GetNbinsX()+1; 
  int MaxEntry=(int)histo->GetMaximum();

  size_t startingSize=mBadStrip.size();
  long double Poisson[MaxEntry+1];
 
  float Nentries=histo->GetEntries();
  int Nbins=histo->GetNbinsX();

  std::map<short,StripStruct>::const_iterator EndIter=mBadStrip.end();
  for(std::map<short,StripStruct>::const_iterator iter=mBadStrip.begin();iter!=EndIter;++iter){
    
    Nentries-=iter->second.N;
    Nbins--;
  }

  if (Nentries==0)
    return;
  
  float meanVal=Nentries/Nbins; 

  //cout << "Iterate " << Nentries << " " << Nbins << " " << meanVal << endl;

  for(int i=0;i<MaxEntry+1;i++){
    Poisson[i]= (i==0)?TMath::Poisson(i,meanVal):Poisson[i-1]+TMath::Poisson(i,meanVal);
  }
  for (Int_t i=ibinStart; i<ibinStop; ++i){
    if (mBadStrip.find(i)==mBadStrip.end()){
      unsigned int pos= (unsigned int)histo->GetBinContent(i);
      if(diff<Poisson[pos] && pos>10){
	
	StripStruct a;
	a.N=histo->GetBinContent(i);
	a.prob=Poisson[pos];
	a.mean=meanVal;
	mBadStrip[i]=a;
      }
    }
  }
  if(mBadStrip.size()!=startingSize)
    iterate(histo,mBadStrip);
}

void badStripStudy(TH1F *histo){
  int Nbads=0;
  int NbadsFirstEdge=0;
  int NbadsSecondEdge=0;

  char cbads[_cbads_LENGTH];
  unsigned long int uno=~0;
  for (size_t i=0;i<_cbads_LENGTH;++i)
    memcpy((void*)&cbads[i],(void*)&uno,1);

  if ( histo->GetEntries() == 0 )
    return;

  std::map<short,StripStruct> mBadStrip;
  
  //  cout << "new" << endl;
  iterate(histo,mBadStrip);

  std::map<short,StripStruct>::const_iterator EndIter=mBadStrip.end();
  for(std::map<short,StripStruct>::const_iterator iter=mBadStrip.begin();iter!=EndIter;++iter){
    
    int i=iter->first;
    std::cout<< "\t" << histo->GetTitle() << " StripNum\t" << i-1 << " \t Entries " << iter->second.N << " \t respect a mean of " << iter->second.mean << " 1-Prob " << scientific << iter->second.prob << fixed <<std::endl;
    
    Nbads++;
    if (i%128 == 0)
      NbadsSecondEdge++;
    if (i%128 == 1)
      NbadsFirstEdge++;
    
    BitAssign(i-1,cbads);
  }

  if(Nbads ){ 
    std::cout<< "&&&&&& " << strstr(histo->GetTitle(),"Det_") <<" \thas "<<Nbads<< " bad strips, on First Edge " << NbadsFirstEdge << " , on Second Edge " << NbadsSecondEdge << " , centrally " << Nbads - NbadsSecondEdge - NbadsFirstEdge << " , out of " << histo->GetNbinsX() << " nEntries " <<  (int) histo->GetEntries() << "\n-----------------------------------------------\n"<< std::endl;
    
    TCanvas C;
    C.SetLogy();
    histo->Draw();
    C.Print(TString(histo->GetTitle())+TString(".gif"));
    
    char title[128];
    sprintf(title,"%s",histo->GetTitle());
    char *ptr=strtok(title,"_");
    int c=0;
    while (ptr!=NULL){
      if (c==2){
	unsigned int detid=atol(ptr);
	Mask = (char *) malloc(_Mask_LENGTH);
	memcpy((void*)Mask,(void*)&detid,4);
	memcpy((void*)&Mask[4],(void*)cbads,_cbads_LENGTH);
	ss.write(Mask,_Mask_LENGTH);
	break;
      }
      ptr=strtok(NULL,"_");
      c++;
    }

    CorrelatePedestals(histo,mBadStrip);
  }  
}

void CorrelatePedestals(TH1F *histo,std::map<short,StripStruct>& mBadStrip){

  char PedHistoName[1024];
  sprintf(PedHistoName,"DBPedestals_%s",&(histo->GetName()[5]));
  //std::cout << "PedHistoName " << PedHistoName << std::endl;
  TH1 *hped = (TH1*)gDirectory->Get( PedHistoName );
  if (hped==NULL)
    return;
  
  int ibinStart= 1; 
  int ibinStop= histo->GetNbinsX()+1;
  const size_t Napvs= histo->GetNbinsX()/128;
  std::vector<short> apvPed[Napvs];
  float apvCM[Napvs];

  int i;
  for (i=ibinStart; i<ibinStop; ++i){
    apvPed[(int)((i-1)/128)].push_back((short)hped->GetBinContent(i));
  }
  
  for (i=0;i<Napvs;++i){
    sort(apvPed[i].begin(),apvPed[i].end());
    apvCM[i]=.5*(apvPed[i][63]+apvPed[i][64]);
  }
  
  std::map<short,StripStruct>::const_iterator EndIter=mBadStrip.end();
  for(std::map<short,StripStruct>::const_iterator iter=mBadStrip.begin();iter!=EndIter;++iter){
    i=iter->first;
    std::cout<< "\tooooo " << hped->GetTitle() << " strip " << i-1 << " apv " << (i-1)/128 << " ped " << hped->GetBinContent(i) << " apvCM " << apvCM[(int)((i-1)/128)] << std::endl;

    CollectivePedHisto->Fill(hped->GetBinContent(i),apvCM[(int)((i-1)/128)]);
  }
  std::cout << "\n--------------------------------------------------\n" << std::endl;
}

void BadStripsFromPosition(char *input, char* output,double prob=1.e-07){
  
  char PedHistoName[1024];
  char inputNoExtention[1024];
  strcat(inputNoExtention,input);
  sprintf(PedHistoName,"%s_HotStrips_PedScatters",strtok(inputNoExtention,"."));  
  CollectivePedHisto = new TH2F("CollectivePedHisto",PedHistoName,512,0.1,2048.1,512,0.1,2048.1);

  f=new TFile(input,"READ"); 
  _prob=prob;

  Navigate();  

  std::cout << "...Creating Collective Histo " << std::endl;

  TCanvas C;
  gStyle->SetOptStat(10);
  C.SetGridx(1);
  C.SetGridy(1);
  CollectivePedHisto->SetMarkerStyle(21);
  CollectivePedHisto->GetXaxis()->SetTitle("Ped Val (ADC)");      
  CollectivePedHisto->GetYaxis()->SetTitle("ApvPedMedian (ADC)");      
  CollectivePedHisto->Draw();
  C.SetLogx();
  C.SetLogy();
  C.Print(TString(CollectivePedHisto->GetTitle())+".gif");
  CollectivePedHisto->Draw("contz");
  C.Print(TString(CollectivePedHisto->GetTitle())+"_cont.gif");
  //cout << "close" << endl;
  //f->Close();     

  std::cout << "....Creating binary file" << std::endl;

  ofstream os;
  os.open (output);
  
  ss.seekg (0, ios::end);
  int length=ss.tellg();
  ss.seekg (0, ios::beg);
  char c[length];
  ss.read(c,length);
  
  os.write(c,length);

  os.close();

  /*
  ifstream is;
  is.open (output, ios::binary );
  is.seekg (0, ios::end);
  length=is.tellg();
  is.seekg (0, ios::beg);
  char cc[length];
  is.read(cc,length);

  for (int i=0;i<length/_Mask_LENGTH;++i){
    unpack(&cc[i*_Mask_LENGTH]);
  }
  */
}

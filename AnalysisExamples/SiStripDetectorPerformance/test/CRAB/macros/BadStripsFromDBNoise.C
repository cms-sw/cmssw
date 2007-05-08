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
#include "TKey.h"
#include "TObject.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TCanvas.h"

char * Mask;
stringstream ss;
TFile *f;
float _LowTh; //sigmas
float _HighTh; //sigmas
float _AbsLowThNoise; //noise
float _AbsHighThNoise; //noise

static const size_t _Mask_LENGTH = 100;
static const size_t _cbads_LENGTH = 96;

struct StripStruct{
  float val;
  long double mean;
  long double sigma;
  bool outAbsRange;
};


void badStripStudy(TH1F *histo);

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

      if (strstr(obj->GetTitle(),"DBNoise_Single")==NULL)
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

void apvStudy(int iApv, TH1F *histo, std::map<short,StripStruct>& mBadStrip){

  int ibinStart= iApv*128+1; //iApv=0 ibinStart=1; iApv=1 ibinStart=129; etc
  int ibinStop= (iApv+1)*128; //iApv=0 ibinStop=128; iApv=1 ibinStop=256; etc
  int Nbin=0;

  float Sbin=0;
  float S2bin=0;
  float Mean=0;
  float Mean2=0;
  float stddev=0;

  for (Int_t i=ibinStart; i<=ibinStop; ++i)
    if (mBadStrip.find(i)==mBadStrip.end()){
      Sbin += histo->GetBinContent(i);
      S2bin += histo->GetBinContent(i)*histo->GetBinContent(i);
      Nbin++;    
    } 
  
  Mean = Sbin/Nbin;
  Mean2 = S2bin/Nbin;
  stddev=sqrt(Mean2-Mean*Mean);

  StripStruct a;
  for (Int_t i=ibinStart; i<=ibinStop; ++i){
    if(histo->GetBinContent(i)>Mean+_HighTh*stddev || histo->GetBinContent(i)<Mean-_LowTh*stddev){
      a.val=histo->GetBinContent(i);
      a.mean=Mean;
      a.sigma=stddev;
      if(histo->GetBinContent(i)>_AbsLowThNoise && histo->GetBinContent(i)<_AbsHighThNoise)
	a.outAbsRange=false;
      else
	a.outAbsRange=true;
      mBadStrip[i]=a;
    }
  }
}

void iterate(TH1F* histo,std::map<short,StripStruct>& mBadStrip){

  size_t startingSize=mBadStrip.size();

  for (int iApv=0;iApv<(Int_t)((histo->GetNbinsX())/128); ++iApv){
    //std::cout << "\n\nAPV i= " << iApv << "  of  "<< (Int_t)((histo->GetNbinsX())/128) << "\n\n" << std::endl;
    apvStudy(iApv,histo,mBadStrip);
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
    std::cout<< "\t" << histo->GetTitle() << " StripNum\t" << i-1 << " \t NoiseVal " << iter->second.val << " \t respect a mean of " << iter->second.mean << " sigma " << iter->second.sigma << " outAbsRange " << iter->second.outAbsRange <<std::endl;
    
    Nbads++;
    if (i%128 == 0)
      NbadsSecondEdge++;
    if (i%128 == 1)
      NbadsFirstEdge++;
    
    BitAssign(i-1,cbads);
  }

  if(Nbads ){ 
    std::cout<< "&&&&&& " << strstr(histo->GetTitle(),"Det_") <<" \thas "<<Nbads<< " bad strips, on First Edge " << NbadsFirstEdge << " , on Second Edge " << NbadsSecondEdge << " , centrally " << Nbads - NbadsSecondEdge - NbadsFirstEdge << " , out of " << histo->GetNbinsX() <<"\n-----------------------------------------------\n"<< std::endl;
    
    TCanvas C;
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
  }  
}


void BadStripsFromDBNoise(char *input, char* output,float LowTh=3, float HighTh=5, float AbsLowThNoise=2, float AbsHighThNoise=6){

  f=new TFile(input,"READ"); 
  _LowTh=LowTh;
  _HighTh=HighTh;
  _AbsLowThNoise=AbsLowThNoise;
  _AbsHighThNoise=AbsHighThNoise;

  Navigate();  

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

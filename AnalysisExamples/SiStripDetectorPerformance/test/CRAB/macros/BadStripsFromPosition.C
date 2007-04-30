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
double _prob;
static const size_t _Mask_LENGTH = 100;
static const size_t _cbads_LENGTH = 96;

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
    char* c;
    int num=(( bitword[4+block] & 0xFF) >> bitshift ) & 0x1;
    unsigned int detid;
    memcpy((void*)&detid,(void*)bitword,4);

    if (num==0)
      std::cout << i << " " << detid << " " << block << " " << bitshift << " unpack " << num << endl;
  }
}

void badStripStudy(TH1F *histo){
  int ibinStart= 1; 
  int ibinStop= histo->GetNbinsX(); 
  int MaxEntry=(int)histo->GetMaximum();
  int Nbads=0;
  int NbadsFirstEdge=0;
  int NbadsSecondEdge=0;
  double diff=1.-_prob; 

  char cbads[_cbads_LENGTH];
  unsigned long int uno=~0;
  for (size_t i=0;i<_cbads_LENGTH;++i)
    memcpy((void*)&cbads[i],(void*)&uno,1);

  float meanVal=histo->GetEntries()/histo->GetNbinsX();

  if ( meanVal == 0 )
    return;

  long double Poisson[MaxEntry+1];

  for(int i=0;i<MaxEntry+1;i++){
    Poisson[i]= (i==0)?TMath::Poisson(i,meanVal):Poisson[i-1]+TMath::Poisson(i,meanVal);
  }
  for (Int_t i=ibinStart; i<ibinStop; ++i)
    {
      unsigned int pos= (unsigned int)histo->GetBinContent(i);
      if(diff<Poisson[pos]){
	std::cout<< "\t" << histo->GetTitle() << " StripNum\t" << i-1 << " \t Entries " << histo->GetBinContent(i) << " \t respect a mean of " << meanVal << " 1-Prob " << scientific << Poisson[pos]<< fixed <<std::endl;
	
	Nbads++;
	if (i%128 == 0)
	  NbadsSecondEdge++;
	if (i%128 == 1)
	  NbadsFirstEdge++;

	BitAssign(i-1,cbads);
      }
      if(Nbads && i==ibinStop-1){ 
	std::cout<< "&&&&&& " << strstr(histo->GetTitle(),"Det_") <<" \thas "<<Nbads<< " bad strips, on First Edge " << NbadsFirstEdge << " , on Second Edge " << NbadsSecondEdge << " , centrally " << Nbads - NbadsSecondEdge - NbadsFirstEdge << " , out of " << i <<"\n-----------------------------------------------\n"<< std::endl;

	TCanvas C;
	histo->Draw();
	C.Print(TString(histo->GetTitle())+TString(".eps"));

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
}


void BadStripsFromPosition(char *input, char* output,double prob=1.e-07){

  f=new TFile(input,"READ"); 
  _prob=prob;

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

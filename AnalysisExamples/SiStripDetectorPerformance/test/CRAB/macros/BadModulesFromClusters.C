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
double _StoNTh, _NsNbRatio; 
static const size_t _Mask_LENGTH = 100;
static const size_t _cbads_LENGTH = 96;

void StoNStudy(TH1F *histo);

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

    if (strstr(obj->GetTitle(),"cStoN_Single")==NULL)
      continue;
    
    TH1F* h = (TH1F*)key->ReadObj();
    
    StoNStudy(h);
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

void StoNStudy(TH1F *histo){

  if ( histo->GetEntries() == 0 ){
    //std::cout << "\n\n WORNING the histo " << histo->GetTitle() << " is empty  " << std::endl;
    return;
  }

  int Ns=0;
  int Nb=0;

  int ibinStart= 0; 
  int ibinStop= histo->GetNbinsX()+2;

  for (int i=ibinStart; i<ibinStop; ++i){

    if (histo->GetBinCenter(i)<_StoNTh) {
      Nb+= (int) histo->GetBinContent(i);
    }else{
      Ns+= (int) histo->GetBinContent(i);
    }
  }

  if ( histo->GetEntries() <100 )
    return;

  if (Nb==0)
    return;

  if (Ns/Nb > _NsNbRatio)
    return;

  char cbads[_cbads_LENGTH];
  unsigned long int zero=0;
  for (size_t i=0;i<_cbads_LENGTH;++i)
    memcpy((void*)&cbads[i],(void*)&zero,1);


  std::cout<< "&&&&&& " << strstr(histo->GetTitle(),"Det_") <<" \thas "<< Ns << " particle entries, " << Nb << " noise entries " << " out of " << histo->GetEntries() <<"\n-----------------------------------------------\n"<< std::endl;
  
  TCanvas C;
  //C.SetLogy();
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

void BadModulesFromClusters(char *input, char* output,double StoNTh=14, double NsNbRatio=1){
  
  f=new TFile(input,"READ"); 
  _StoNTh=StoNTh;
  _NsNbRatio=NsNbRatio;

  Navigate();  

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
}

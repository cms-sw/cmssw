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
double _pedTh;
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

    if (strstr(obj->GetTitle(),"DBPedestals_Single")==NULL)
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

void iterate(TH1F* histo,std::map<short,double>& mBadApv){
  
  int ibinStart= 1; 
  int ibinStop= histo->GetNbinsX()+1;
  const size_t Napvs= histo->GetNbinsX()/128;
  std::vector<short> apvPed[Napvs];
  float apvCM[Napvs];
  bool apvGood[Napvs];
    
  int i;

  for (i=0;i<Napvs;++i)
    apvGood[i]=true;

  for (i=ibinStart; i<ibinStop; ++i){
    apvPed[(int)((i-1)/128)].push_back((short)histo->GetBinContent(i));
    if (histo->GetBinContent(i)>_pedTh)
      apvGood[(int)((i-1)/128)]=false;
  }
  
  for (i=0;i<Napvs;++i){
    sort(apvPed[i].begin(),apvPed[i].end());
    apvCM[i]=.5*(apvPed[i][63]+apvPed[i][64]);
    if (!apvGood[i])
      mBadApv[i]=apvCM[i];
  }
}

void badStripStudy(TH1F *histo){
  int NAPVbads=0;

  char cbads[_cbads_LENGTH];
  unsigned long int uno=~0;
  for (size_t i=0;i<_cbads_LENGTH;++i)
    memcpy((void*)&cbads[i],(void*)&uno,1);

  if ( histo->GetEntries() == 0 ){
    std::cout << "\n\n WORNING the histo " << histo->GetTitle() << " is empty  " << std::endl;
    return;
  }

  std::map<short,double> mBadApv;
  
  //  cout << "new" << endl;
  iterate(histo,mBadApv);

  std::map<short,double>::const_iterator EndIter=mBadApv.end();
  for(std::map<short,double>::const_iterator iter=mBadApv.begin();iter!=EndIter;++iter){
    
    std::cout<< "\t" << histo->GetTitle() << " APV\t" << iter->first << " \t PedMedian " << iter->second << std::endl;
    NAPVbads++;

    for (int i=iter->first*128;i<(iter->first+1)*128;++i)
      BitAssign(i,cbads);
  }

  if(NAPVbads ){ 
    std::cout<< "&&&&&& " << strstr(histo->GetTitle(),"Det_") <<" \thas "<<NAPVbads<< " out of " << histo->GetNbinsX()/128 <<"\n-----------------------------------------------\n"<< std::endl;
    
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
  }  
}

void BadModulesFromPedestals(char *input, char* output,double pedTh=768){
  
  f=new TFile(input,"READ"); 
  _pedTh=pedTh;

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

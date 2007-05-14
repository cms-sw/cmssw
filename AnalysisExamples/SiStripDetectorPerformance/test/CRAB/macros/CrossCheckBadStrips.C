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

static const int debug=1;
char * Mask;
stringstream ss;
TFile *f;
size_t block;
size_t bitshift;
static const size_t _Mask_LENGTH = 100;
static const size_t _cbads_LENGTH = 96;

std::map<unsigned int,const char *> BadStripsMap[2];
char* BadStripBlob[2];

void Compare(std::map<unsigned int,const char *> A,std::map<unsigned int,const char *> B){

  int count=0;
  int countStrip=0;
  int countNoIn=0;

  std::map<unsigned int,const char*>::const_iterator EnditerB=B.end();

  for (std::map<unsigned int,const char*>::const_iterator iterA=A.begin();iterA!=A.end();++iterA){
    std::map<unsigned int,const char*>::const_iterator iterB=B.find(iterA->first);
    if (iterB!=EnditerB){
      for (size_t i=0;i<768;++i){
	block=int(i/8);
	bitshift=i%8;
	int numA=(( iterA->second[block] & 0xFF) >> bitshift ) & 0x1;
	int numB=(( iterB->second[block] & 0xFF) >> bitshift ) & 0x1;
	bool first=true;
	if (numA!=numB && numA==0){
	  countStrip++;
	  if(first){
	    count++;
	    cout << "\nDet " << iterA->first << " strips in A not in B ";
	  }
	  first=false;
	  std::cout << " " << i << "\t" ;//<< numA << " " << numB << endl;
	}
      }
      //      cout << endl;
    }else{
      count++;
      countNoIn++;
      cout << "\nDet " << iterA->first << " in A not in B ";
	for (size_t i=0;i<768;++i){
	  block=int(i/8);
	  bitshift=i%8;
	  int num=(( iterA->second[block] & 0xFF) >> bitshift ) & 0x1;
	  if (num==0){
	    countStrip++;
	    std::cout << "\t " << i ;
	  }
	}
	//cout << endl;
    }
  }

  cout << "\n\nN Bad modules not matching " << countNoIn << endl;
  cout << "N Bad modules with strips not matching " << count << endl;
  cout << "N Bad strips not matching " << countStrip << endl;
}

void setBadStripsMap(const char* bitword,size_t k){

  unsigned int detid;
  
  memcpy((void*)&detid,(void*)bitword,4);
  BadStripsMap[k][detid]=&bitword[4];

  if (debug>2)
    for (size_t i=0;i<768;++i){
      block=int(i/8);
      bitshift=i%8;
      int num=(( bitword[4+block] & 0xFF) >> bitshift ) & 0x1;
      if (num==0)
	std::cout << i << " " << detid << " " << block << " " << bitshift << " setBadStripsMap " << num << endl;
    }
}
void setBadStripsFile(char* name, size_t k){
  cout << "\nLoad " << name << endl;
  ifstream is;
  is.open(name, ios::binary );
  if (!is.good())
    return;
  is.seekg (0, ios::end);
  int length=is.tellg();
  is.seekg (0, ios::beg);
  BadStripBlob[k] = (char*) malloc(length);
  is.read(BadStripBlob[k],length);
  
  for (size_t i=0;i<length/_Mask_LENGTH;++i){
    setBadStripsMap(&(BadStripBlob[k])[i*_Mask_LENGTH],k);
  }
  is.close();

  std::cout << "\nList of BadStrips " << std::endl;
  for (std::map<unsigned int,const char*>::const_iterator iter=BadStripsMap[k].begin();iter!=BadStripsMap[k].end();++iter){
    for (size_t i=0;i<768;++i){
      block=int(i/8);
      bitshift=i%8;
      int num=(( iter->second[block] & 0xFF) >> bitshift ) & 0x1;
      if (num==0)
	std::cout << "det " << iter->first << " strip " << i << " -- "  << block << " " << bitshift << " setBadStripsMap " << num << std::endl;
    }    
  }
}

void CrossCheckBadStrips(char* input1, char* input2){


  setBadStripsFile(input1,0);
  cout << "\n--------------------------" << endl;
  setBadStripsFile(input2,1);
  cout << "\n--------------------------" << endl;
  cout << "A = " << input1 << " B = " << input2 << endl;
  Compare(BadStripsMap[0],BadStripsMap[1]);
  cout << "\n--------------------------" << endl;
  cout << "A = " << input2 << " B = " << input1 << endl;
  Compare(BadStripsMap[1],BadStripsMap[0]);
  cout << "\n--------------------------" << endl;
}

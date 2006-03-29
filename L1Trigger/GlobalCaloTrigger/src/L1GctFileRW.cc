#include "L1Trigger/GlobalCaloTrigger/interface/RctFileRW.h"
#include <iostream>

using std::vector;
typedef vector<unsigned> dataVector;

RctFileRW::RctFileRW(char inputFile[256]): theData(64){
  file.open(inputFile,ios::in);
//  _bx="1";
//  _run=1;
//  if(!file){
//    cout << "Cannot open input data file" << endl;
//    exit(1);
//  }
//  file>>_bx>>_run;  
} 

RctFileRW::~RctFileRW(){
  file.close();
}

void RctFileRW::readRctFile(){
  cout << "Reading bunch Crossing " << _run << endl;
  unsigned candidate;
  
 //loop over bunch crossings (j)   
 for(int j=0;j<2;j++){
// RctCableData RctCableObject;
//    
//  //Read in Electrons: i<4 is iso, i = 4-8 is non-iso
//   for(int i=0;i<8;i++){
//     file >> std::hex >> candidate;
//     if(i<4){
//       RctCableObject.setIsoElectron(i, candidate);
//     }
//     else{
//       RctCableObject.setNonIsoElectron((i-4), candidate);
//     }
//   }
// 
//  for(int i=0;i<14;i++){
//    file >> std::hex >> candidate;
//    RctCableObject.setMipBit(i, candidate); 
//   }
//  for(int i=0;i<14;i++){
//    file >> std::hex >> candidate; 
//    RctCableObject.setQuietBit(i, candidate);
//  }
//  //run through the jet stuff and HF to be ready to read in next event
//  for(int i=0;i<22;i++){
//    file >> std::hex >> candidate;
//  }
//
//  theData[j]=RctCableObject;
//  
//  file >> _bx >> _run;
//  cout << "Reading bunch "<<_bx<<" "<<_run<<endl;
 }

 file >> _bx >> _run;
}

void RctFileRW::writeRctFile(){
  this->readRctFile();
  ofile.open("GctRctFile.txt",ios::out);
  //looping over bx's
//  for(int j=0;j<2;j++){ 
//    ofile<<"Crossing ";
//    ofile<<j;
//    ofile<<"\n";
//    for(int i=0;i<4;i++){
//      ofile<<std::hex<<theData[j].getIsoElectron(i);
//      ofile<<" ";
//    }
//    for(int i=0;i<4;i++){
//      ofile<<std::hex<<theData[j].getNonIsoElectron(i);
//      ofile<<" ";
//    }
//    ofile<<"\n";
//    for(int i=0;i<14;i++){
//      ofile<<std::hex<<theData[j].getMipBit(i);
//      ofile<<" ";
//    }
//    ofile<<"\n";
//    for(int i=0;i<14;i++){
//      ofile<<std::hex<<theData[j].getQuietBit(i);
//      ofile<<" ";
//    }
//    ofile<<"\n";
//    for(int i=0;i<22;i++){
//      ofile<<"na";
//      ofile<<" ";
//    }
//    ofile<<"\n";
//  }
  ofile.close();
 }






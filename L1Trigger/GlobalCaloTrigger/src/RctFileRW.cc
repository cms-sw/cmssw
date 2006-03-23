#include "L1Trigger/GlobalCaloTrigger/interface/RctFileRW.h"
#include <iostream>

using namespace std;
typedef vector<unsigned> dataVector;
typedef vector<RctCableData> RctCrateData;

RctFileRW::RctFileRW(char inputFile[256]): theData(64){
  file.open(inputFile,ios::in);
  _bx="1";
  _run=1;
  if(!file){
    cout << "Cannot open input data file" << endl;
    exit(1);
  }
  file>>_bx>>_run;  
} 

RctFileRW::~RctFileRW(){
  file.close();
}

void RctFileRW::readRctFile(){
  cout << "Reading bunch Crossing " << _run << endl;
  unsigned candidate;
  
 //loop over bunch crossings (j)   
 for(int j=0;j<2;j++){
 RctCableData RctCableObject;
    
  //Read in Electrons: i<4 is iso, i = 4-8 is non-iso
   for(int i=0;i<8;i++){
     file >> std::hex >> candidate;
     if(i<4){
       RctCableObject.setIsoElectron(i, candidate);
     }
     else{
       RctCableObject.setNonIsoElectron((i-4), candidate);
     }
   }
 
  for(int i=0;i<14;i++){
    file >> std::hex >> candidate;
    RctCableObject.setMipBit(i, candidate); 
   }
  for(int i=0;i<14;i++){
    file >> std::hex >> candidate; 
    RctCableObject.setQuietBit(i, candidate);
  }
  //run through the jet stuff and HF to be ready to read in next event
  for(int i=0;i<22;i++){
    file >> std::hex >> candidate;
  }

  theData[j]=RctCableObject;
  
  file >> _bx >> _run;
  cout << "Reading bunch "<<_bx<<" "<<_run<<endl;
 }

 file >> _bx >> _run;
}

void RctFileRW::writeRctFile(){
  this->readRctFile();
  ofile.open("GctRctFile.txt",ios::out);
  //looping over bx's
  for(int j=0;j<2;j++){ 
    ofile<<"Crossing ";
    ofile<<j;
    ofile<<"\n";
    for(int i=0;i<4;i++){
      ofile<<std::hex<<theData[j].getIsoElectron(i);
      ofile<<" ";
    }
    for(int i=0;i<4;i++){
      ofile<<std::hex<<theData[j].getNonIsoElectron(i);
      ofile<<" ";
    }
    ofile<<"\n";
    for(int i=0;i<14;i++){
      ofile<<std::hex<<theData[j].getMipBit(i);
      ofile<<" ";
    }
    ofile<<"\n";
    for(int i=0;i<14;i++){
      ofile<<std::hex<<theData[j].getQuietBit(i);
      ofile<<" ";
    }
    ofile<<"\n";
    for(int i=0;i<22;i++){
      ofile<<"na";
      ofile<<" ";
    }
    ofile<<"\n";
  }
  ofile.close();
 }


RctCrateData RctFileRW::getData(){
  RctCrateData AllCrossings; 
  this->readRctFile();
  AllCrossings = theData;
  return AllCrossings;
}

void RctFileRW::setData(RctCrateData data){
unsigned candidate;
  
 //loop over RctCableData size (j)   
 for(int j=0;j<data.size();j++){
   RctCableData RctCableObject;
   
   //Read in Electrons: i<4 is iso, i = 4-8 is non-iso
   for(int i=0;i<8;i++){
     candidate = data[j].getIsoElectron(i);
     if(i<4){
       RctCableObject.setIsoElectron(i, candidate);
     }
     else{
       RctCableObject.setNonIsoElectron((i-4), candidate);
     }
   } 
   for(int i=0;i<14;i++){
     candidate = data[j].getMipBit(i);
     RctCableObject.setMipBit(i, candidate); 
   }
   for(int i=0;i<14;i++){
     candidate = data[j].getQuietBit(i); 
     RctCableObject.setQuietBit(i, candidate);
   }
   theData[j]=RctCableObject;
 }
}

Block RctFileRW::getCable(int nCable){
  Block cable;
  //  for(int i=0;i<theData.size();i++){ !!!OBS commented out for now, cos it's 64
  for(int i=0;i<2;i++){
    long long test = theData[i].getCable(nCable);
    unsigned long cycle0 = (theData[i].getCable(nCable))&(0xffffffff);
    unsigned long cycle1 = ((theData[i].getCable(nCable))>>32)&(0xffffffff);
    cable.pushTopWord32(cycle0);
    cable.pushTopWord32(cycle1);
  }
  return cable;
}

void RctFileRW::setCable(Block data){
  unsigned long long input;
  unsigned long cycle, size;
  size = data.sizeIn32BitWords();
  for(int i=0;i<(size/2);i++){
    for(int j=0;j<(size/12);j++){
      cycle = data.getWord32((i*2));
      input = (cycle<<31)&(0x00000000ffffffff);
      cycle = data.getWord32((i*2)+1);
      input = cycle&(0xffffffff00000000);
      theData[j].setCable(i,input);
    }
  }
}

void RctFileRW::writeCableFile(Block dataBlock){
  unsigned int size = dataBlock.sizeIn32BitWords();
  ofile.open("GctCableFile.txt");
  //looping over bx's
  for(int j=0;j<2;j++){ 
    ofile<<"Crossing ";
    ofile<<j;
    ofile<<"\n";
    for(int i=0;i<size;i++){
     ofile<<std::hex<<dataBlock.getWord32(i);
      ofile<<"\n";
    }
  }
  ofile.close();
}


vector<unsigned> RctFileRW::getIsoElectrons(int bx){
  vector<unsigned> theIsoElectrons;
  this->readRctFile();
  if(bx==66){ 
    for(int j=0;j<64;j++){
      for(int m=0;m<4;m++){
	theIsoElectrons.push_back(theData[j].getIsoElectron(m));
      }
    }
  }else{
    for(int m=0;m<4;m++){
      theIsoElectrons.push_back(theData[bx].getIsoElectron(m));
    }
  }
  return theIsoElectrons;
}

vector<unsigned> RctFileRW::getNonIsoElectrons(int bx){
  vector<unsigned> theNonIsoElectrons;
  this->readRctFile();
  if(bx==66){ 
    for(int j=0;j<64;j++){
      for(int m=0;m<4;m++){
	theNonIsoElectrons.push_back(theData[j].getNonIsoElectron(m));
      }
    }
  }else{
    for(int m=0;m<4;m++){
      theNonIsoElectrons.push_back(theData[bx].getNonIsoElectron(m));
    }
  }
  
  return theNonIsoElectrons;
}

vector<unsigned> RctFileRW::getMipBits(int bx){
  vector<unsigned> theMipBits;
  this->readRctFile();
  if(bx==66){ 
    for(int j=0;j<64;j++){
      for(int m=0;m<4;m++){
	theMipBits.push_back(theData[j].getMipBit(m));
      }
    }
  }else{
    for(int m=0;m<4;m++){
      theMipBits.push_back(theData[bx].getMipBit(m));
    }
  }
  
  return theMipBits;
}

vector<unsigned> RctFileRW::getQuietBits(int bx){
  vector<unsigned> theQuietBits;
  this->readRctFile();
  if(bx==66){ 
    for(int j=0;j<64;j++){
      for(int m=0;m<4;m++){
	theQuietBits.push_back(theData[j].getQuietBit(m));
      }
    }
  }else{
    for(int m=0;m<4;m++){
      theQuietBits.push_back(theData[bx].getQuietBit(m));
    }
  }
  
  return theQuietBits;
}




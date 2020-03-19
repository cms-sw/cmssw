// This class holds all the stubs in a DTC region for a give layer
#ifndef ALLSTUBSMEMORY_H
#define ALLSTUBSMEMORY_H

#include "L1TStub.h"
#include "Stub.h"
#include "MemoryBase.h"

#include <ctype.h>

using namespace std;

class AllStubsMemory:public MemoryBase{

public:

  AllStubsMemory(string name, unsigned int iSector, 
	       double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;

    //set the layer or disk that the memory is in
    initLayerDisk(3,layer_,disk_);

    assert(name.substr(5,3)=="PHI");
  }

  void addStub(std::pair<Stub*,L1TStub*> stub) {
    stubs_.push_back(stub);
  }

  unsigned int nStubs() const {return stubs_.size();}

  Stub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<Stub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  void clean() {
    stubs_.clear();
  }

  void writeStubs(bool first) {

    openFile(first,"../data/MemPrints/Stubs/AllStubs_");
    
    for (unsigned int j=0;j<stubs_.size();j++){
      string stub=stubs_[j].first->str();
      out_ << "0x";
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ <<" "<<stub << " " <<hexFormat(stub)<<endl;
    }
    out_.close();
  }

  int layer() const { return layer_;}
  int disk() const { return disk_;}

private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<Stub*,L1TStub*> > stubs_;

  int layer_;
  int disk_;

};

#endif

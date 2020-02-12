// This class holds the list of candidate stub pairs 
#ifndef STUBPAIRSMEMORY_H
#define STUBPAIRSMEMORY_H

#include "L1TStub.h"
#include "Stub.h"
#include "VMStubTE.h"
#include "MemoryBase.h"

using namespace std;

class StubPairsMemory:public MemoryBase{

public:

  StubPairsMemory(string name, unsigned int iSector, 
		double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addStubPair(const VMStubTE& stub1,
		   const VMStubTE& stub2,
                   const unsigned index = 0,
                   const std::string &tedName = "") {
    stubs1_.push_back(stub1);
    stubs2_.push_back(stub2);
    indices_.push_back(index);
    tedNames_.push_back(tedName);
  }

  unsigned int nStubPairs() const {return stubs1_.size();}

  VMStubTE getVMStub1(unsigned int i) const {return stubs1_[i];}
  Stub* getFPGAStub1(unsigned int i) const {return stubs1_[i].stub().first;}
  L1TStub* getL1TStub1(unsigned int i) const {return stubs1_[i].stub().second;}
  std::pair<Stub*,L1TStub*> getStub1(unsigned int i) const {return stubs1_[i].stub();}

  VMStubTE getVMStub2(unsigned int i) const {return stubs2_[i];}
  Stub* getFPGAStub2(unsigned int i) const {return stubs2_[i].stub().first;}
  L1TStub* getL1TStub2(unsigned int i) const {return stubs2_[i].stub().second;}
  std::pair<Stub*,L1TStub*> getStub2(unsigned int i) const {return stubs2_[i].stub();}

  unsigned getIndex(const unsigned i) const {return indices_.at(i);}
  const std::string &getTEDName(const unsigned i) const {return tedNames_.at(i);}

  void clean() {
    stubs1_.clear();
    stubs2_.clear();
    indices_.clear();
    tedNames_.clear();
  }

  void writeSP(bool first) {

    std::string fname="../data/MemPrints/StubPairs/StubPairs_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    for (unsigned int j=0;j<stubs1_.size();j++){
      string stub1index=stubs1_[j].stub().first->stubindex().str();
      string stub2index=stubs2_[j].stub().first->stubindex().str();
      out_ << "0x";
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ <<" "<<stub1index <<"|"<<stub2index << " " <<hexFormat(stub1index+stub2index)<<endl;
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }


private:

  double phimin_;
  double phimax_;
  //FIXME should not be two vectors
  std::vector<VMStubTE> stubs1_;
  std::vector<VMStubTE> stubs2_;

  std::vector<unsigned> indices_;
  std::vector<std::string> tedNames_;

};

#endif

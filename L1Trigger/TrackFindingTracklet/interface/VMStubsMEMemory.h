//This class holds the reduced VM stubs
#ifndef VMSTUBSMEMEMORY_H
#define VMSTUBSMEMEMORY_H

#include "L1TStub.h"
#include "Stub.h"
#include "MemoryBase.h"

using namespace std;

class VMStubsMEMemory:public MemoryBase{

public:

  VMStubsMEMemory(string name, unsigned int iSector, 
	      double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addStub(std::pair<Stub*,L1TStub*> stub) {
    stubs_.push_back(stub);
    if (stub.first->isBarrel()) { // barrel
      int bin=(1<<(MEBinsBits-1))+(stub.first->z().value()>>(stub.first->z().nbits()-MEBinsBits));
      //cout << "VMStubsMEMemory::addStub "<<bin<<" "<<stub.first->z().value()<<" "<<stub.first->z().nbits()<<endl;
      assert(bin>=0);
      assert(bin<MEBins);
      if (debug1) {
	cout << getName() << " adding stub to bin "<<bin<<endl;
      }
      binnedstubs_[bin].push_back(stub);
    }
    else { // disk 
      int ir = stub.first->r().value();
      //presumably the VMRouter can use a lookup table for the bin.
      //For now implement a simple cut
      double rstub=ir*kr;
      if (!stub.first->isPSmodule()){
	assert(ir<10);
	if (abs(stub.first->disk().value())<=2) {
	  rstub=rDSSinner[ir];
	} else {
	  rstub=rDSSouter[ir];
	}
      }
      int bin=8.0*(rstub-rmindiskvm)/(rmaxdisk-rmindiskvm);
      assert(bin>=0);
      assert(bin<MEBinsDisks);
      if (stub.first->disk().value()<0) bin+=MEBinsDisks;
      if (debug1) {
	cout << getName() << " adding stub to bin "<<bin<<endl;
      }
      binnedstubs_[bin].push_back(stub);
      
    }
  }

  unsigned int nStubs() const {return stubs_.size();}

  Stub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<Stub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  unsigned int nStubsBin(unsigned int bin) const {
    //if (layer_>0){
    //  assert(bin<MEBins);
    // }else {
      assert(bin<MEBinsDisks*2);
      //}
    return binnedstubs_[bin].size();
  }

  std::pair<Stub*,L1TStub*> getStubBin(unsigned int bin, unsigned int i) const {
    //if (layer_>0){
    //  assert(bin<MEBins);
    //}else {
      assert(bin<MEBinsDisks*2);
      //}
    assert(i<binnedstubs_[bin].size());
    return binnedstubs_[bin][i];
  }
  
  void clean() {
    stubs_.clear();
    for (unsigned int i=0; i<MEBinsDisks*2; i++){
      binnedstubs_[i].clear();
    }
  }

  void writeStubs(bool first) {

    std::string fname="../data/MemPrints/VMStubsME/VMStubs_";
    fname+=getName();
    //get rid of duplicates
    int len = fname.size();
    if(fname[len-2]=='n'&& fname[len-1]>'1'&&fname[len-1]<='9') return;
    //
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_ = 0;
      event_ = 1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    

    
    for (unsigned int i=0;i<NLONGVMBINS;i++) {
      for (unsigned int j=0; j<binnedstubs_[i].size();j++) {
        string stub = binnedstubs_[i][j].first->stubindex().str();
	stub +=  "|" + binnedstubs_[i][j].first->bend().str();

	FPGAWord finepos=binnedstubs_[i][j].first->finez();
	if (!binnedstubs_[i][j].first->isBarrel()) {
	  finepos=binnedstubs_[i][j].first->finer();
	}
	stub +=  "|" + finepos.str();
        out_ << hex << i << " " << j << dec << " " << stub << " " <<hexFormat(stub)<<endl;
      }
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }

private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<Stub*,L1TStub*> > stubs_;

  std::vector<std::pair<Stub*,L1TStub*> > binnedstubs_[MEBinsDisks*2];

  
  
};

#endif

// This class holds the list of candidate stub pairs 
#ifndef FPGASTUBTRIPLETS_H
#define FPGASTUBTRIPLETS_H

#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGAMemoryBase.hh"

using namespace std;

class FPGAStubTriplets:public FPGAMemoryBase{

public:

  FPGAStubTriplets(string name, unsigned int iSector, 
		double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addStubs(std::pair<FPGAStub*,L1TStub*> stub1,
		std::pair<FPGAStub*,L1TStub*> stub2,
		std::pair<FPGAStub*,L1TStub*> stub3) {
    stubs1_.push_back(stub1);
    stubs2_.push_back(stub2);
    stubs3_.push_back(stub3);
  }

  unsigned int nStubTriplets() const {return stubs1_.size();}

  FPGAStub* getFPGAStub1(unsigned int i) const {return stubs1_[i].first;}
  L1TStub* getL1TStub1(unsigned int i) const {return stubs1_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub1(unsigned int i) const {return stubs1_[i];}

  FPGAStub* getFPGAStub2(unsigned int i) const {return stubs2_[i].first;}
  L1TStub* getL1TStub2(unsigned int i) const {return stubs2_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub2(unsigned int i) const {return stubs2_[i];}

  FPGAStub* getFPGAStub3(unsigned int i) const {return stubs3_[i].first;}
  L1TStub* getL1TStub3(unsigned int i) const {return stubs3_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub3(unsigned int i) const {return stubs3_[i];}

  void clean() {
    stubs1_.clear();
    stubs2_.clear();
    stubs3_.clear();
  }


  void writeST(bool first) {

    std::string fname="../data/MemPrints/StubPairs/StubTriplets_";
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
      //string stub1index=stubs1_[j].first->stubindex().str();
      //string stub2index=stubs2_[j].first->stubindex().str();
      //string stub3index=stubs3_[j].first->stubindex().str();
      string stub1index=stubs1_[j].first->stubaddressaste().str();
      string stub2index=stubs2_[j].first->stubaddressaste().str();
      string stub3index=stubs3_[j].first->stubaddressaste().str();
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ <<" "<<stub1index <<"|"<<stub2index << "|" << stub3index << endl;
	  /*
      if(writeoutReal){
	out_ <<" "<<stubs1_[j].first->str_phys() <<" "<<stubs2_[j].first->str_phys()<< endl;
      }else{
	out_ <<" "<<stubs1_[j].first->stubindex().str() <<" "<<stubs2_[j].first->stubindex().str()<< endl;
	//out_ <<" "<<stubs1_[j].first->strbare() <<" "<<stubs2_[j].first->strbare()<< endl;
      }
	  */
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }


private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs1_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs2_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs3_;

};

#endif

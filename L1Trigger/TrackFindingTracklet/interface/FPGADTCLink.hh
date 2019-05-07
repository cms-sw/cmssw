// This class holds a list of stubs that are in a given layer and DCT region
#ifndef FPGADTCLINK_H
#define FPGADTCLINK_H

#include "L1TStub.hh"
#include "FPGAStub.hh"

using namespace std;

class FPGADTCLink{

public:

  FPGADTCLink(double phimin, double phimax){
    if (phimin>M_PI) {
      phimin-=2*M_PI;
      phimax-=2*M_PI;
    }
    assert(phimax>phimin);
    phimin_=phimin;
    phimax_=phimax;
  }

  void addStub(std::pair<FPGAStub*,L1TStub*> stub) {
    stubs_.push_back(stub);
  }

  bool inRange(double phi,bool overlaplayer){
    //cout << "FPGADTCLink::inRange "<<phi<<" "<<phimin_<<" "<<phimax_<<endl;
    double phimax=phimax_;
    double phimin=phimin_;
    if (overlaplayer) {
      double dphi=phimax-phimin;
      assert(dphi>0.0);
      assert(dphi<M_PI);
      phimin-=dphi/6.0;
      phimax+=dphi/6.0;
    }
    //cout << "FPGADTCLink::inRange phi : "<<phi
    // 	 <<" phi min max "<<phimin<<" "<<phimax<<endl;
    return (phi<phimax&&phi>phimin)||(phi+2*M_PI<phimax&&phi+2*M_PI>phimin);
  }
  
  unsigned int nStubs() const {return stubs_.size();}

  FPGAStub* getFPGAStub(unsigned int i) const {return stubs_[i].first;}
  L1TStub* getL1TStub(unsigned int i) const {return stubs_[i].second;}
  std::pair<FPGAStub*,L1TStub*> getStub(unsigned int i) const {return stubs_[i];}

  void clean() {
    stubs_.clear();
  }


private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<FPGAStub*,L1TStub*> > stubs_;



};

#endif

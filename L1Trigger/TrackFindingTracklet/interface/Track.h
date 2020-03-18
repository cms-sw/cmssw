#ifndef FPGATRACK_HH
#define FPGATRACK_HH

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <map>

#include "Util.h"

using namespace std;

class Track{

public:

  Track(int irinv, int iphi0, int id0, int it, int iz0, int ichisq,
            double chisq,
            std::map<int, int> stubID, std::vector<L1TStub*> l1stub,
            int seed){

    irinv_=irinv;
    iphi0_=iphi0;
    id0_=id0;
    iz0_=iz0;
    it_=it;
    ichisq_=ichisq;

    chisq_=chisq;

    nstubs_=l1stub.size();
    if (nstubs_>6) nstubs_=6; //maximum used in fit
    
    //assert(stubID.size()==nstubs_);
    
    stubID_=stubID;
    l1stub_=l1stub;

    seed_=seed;
    duplicate_=false;
    sector_=NSector;

  }

  ~Track() {

  }
  
  void setDuplicate(bool flag) { duplicate_=flag; }
  void setSector(int nsec) { sector_=nsec; }
  void setStubIDpremerge(std::vector<std::pair<int, int>> stubIDpremerge) { stubIDpremerge_ = stubIDpremerge; }
  void setStubIDprefit(std::vector<std::pair<int, int>> stubIDprefit) { stubIDprefit_ = stubIDprefit; }


  int irinv() const { return irinv_; }
  int iphi0() const { return iphi0_; }
  int id0() const { return id0_; }
  int iz0()   const { return iz0_; }
  int it()    const { return it_; }
  int ichisq() const {return ichisq_;}

  std::map<int, int> stubID() const { return stubID_; }
  std::vector<L1TStub*> stubs() const { return l1stub_; }
  std::vector<std::pair<int, int>> stubIDpremerge() const { return stubIDpremerge_; }
  std::vector<std::pair<int, int>> stubIDprefit() const { return stubIDprefit_; }
  

  int seed() const { return seed_; }
  int duplicate() const { return duplicate_; }
  int sector() const { return sector_; }
  
  double pt(double bfield=3.811202) const {
    return (0.3*bfield/100.0)/(irinv_*krinvpars);
  }
  double phi0() const {

    double dphi=2*M_PI/NSector;
    double dphiHG=0.5*dphisectorHG-M_PI/NSector;
    double phimin=sector_*dphi-dphiHG;
    double phimax=phimin+dphi+2*dphiHG;
    phimin-=M_PI/NSector;
    phimax-=M_PI/NSector;
    phimin=Util::phiRange(phimin);
    phimax=Util::phiRange(phimax);
    if (phimin>phimax)  phimin-=2*M_PI;
    double phioffset=phimin;
  
    return iphi0_*kphi0pars+phioffset;
  }
  double eta() const {
    return asinh(it_*ktpars);
  }
  double tanL() const {
    return it_*ktpars;
  }
  double z0() const {
    return iz0_*kz0pars;
  }
  double rinv() const {
    return irinv_*krinvpars;
  }
  double d0() const {return id0_*kd0pars;} //Fix when fit for 5 pars
  double chisq() const {return chisq_;}

  int nPSstubs() const {
    int npsstubs=0;
    for (unsigned int i=0;i<l1stub_.size();i++){
      if (l1stub_[i]->layer()<3) npsstubs++;
    }
    return npsstubs;
  }
  
private:
  
  int irinv_;
  int iphi0_;
  int id0_;  
  int iz0_;
  int it_;
  int ichisq_;

  double rinv_;
  double phi0_;
  double d0_;
  double z0_;
  double t_;
  double chisq_;

  std::vector<std::pair<int, int>> stubIDpremerge_;
  std::vector<std::pair<int, int>> stubIDprefit_;
  std::map<int, int> stubID_;
  std::vector<L1TStub*> l1stub_;

  unsigned int nstubs_;
  int seed_;
  bool duplicate_;
  int sector_;

};

#endif




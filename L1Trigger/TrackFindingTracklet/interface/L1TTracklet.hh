#ifndef L1TTRACKLET_H
#define L1TTRACKLET_H

#include <iostream>
#include <fstream>
#include <map>
#include <assert.h>

using namespace std;


class L1TTracklet{

public:

  L1TTracklet(double rinv, double phi0, double t, double z0){
    
    rinv_=rinv;
    phi0_=phi0;
    t_=t;
    z0_=z0;

  }

  L1TTracklet(){
  }

  void addStub(const L1TStub& j){
    stubs_.push_back(j);
  }

  vector<L1TStub> getStubs() const {
    return stubs_;
  }

  int tpid(double& fraction) const {


    map<int, int> tpids;

    for(unsigned int i=0;i<stubs_.size();i++){
      vector<int> tps=stubs_[i].tps();
      for(unsigned int j=0;j<tps.size();j++){
	tpids[tps[i]]++;
      }
    }

    int tpid=-1;
    int ntps=-1;

    map<int, int>::const_iterator it=tpids.begin();

    while(it!=tpids.end()) {
      if (it->second>ntps) {
	ntps=it->second;
	tpid=it->first;
      }
      it++;
    }

    fraction=(1.0*ntps)/stubs_.size();

    return tpid;

  }

  double pt(double bfield) const { return 0.00299792*bfield/rinv_; }

  int nStubs() const {return stubs_.size();}
  
  double rinv() const {return rinv_;}
  double phi0() const {return phi0_;}
  double t() const {return t_;}
  double z0() const {return z0_;}

  double r() const {return stubs_[0].r();}
  double z() const {return stubs_[0].z();}


private:

  double rinv_;
  double phi0_;
  double t_;
  double z0_;

  vector<L1TStub> stubs_;

};



#endif




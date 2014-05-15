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

  int simtrackid(double& fraction) const {


    map<int, int> simtrackids;

    for(unsigned int i=0;i<stubs_.size();i++){
      //cout << "Stub simtrackid="<<stubs_[i].simtrackid()<<endl;
      simtrackids[stubs_[i].simtrackid()]++;
    }

    int simtrackid=0;
    int nsimtrack=0;

    map<int, int>::const_iterator it=simtrackids.begin();

    while(it!=simtrackids.end()) {
      //cout << it->first<<" "<<it->second<<endl;
      if (it->second>nsimtrack) {
	nsimtrack=it->second;
	simtrackid=it->first;
      }
      it++;
    }

    fraction=(1.0*nsimtrack)/stubs_.size();

    return simtrackid;

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




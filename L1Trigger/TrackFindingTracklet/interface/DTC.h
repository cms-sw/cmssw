// This class holds a list of stubs that are in a given layer and DCT region
#ifndef DTC_H
#define DTC_H

#include "L1TStub.h"
#include "Stub.h"
#include "DTCLink.h"

using namespace std;

class DTC{

public:

  DTC(string name=""){
    name_=name;
    for (unsigned int i=0;i<11;i++) {
      phimin_[i]=10.0;
      phimax_[i]=-10.0;
    }
  }

  void init(string name){
    name_=name;
  }

  void addSec(int sector){
    sectors_.push_back(sector);
  }

  void addphi(double phi,int layerdisk){

    assert(layerdisk>=0);
    assert(layerdisk<11);
    if (phi<phimin_[layerdisk]) phimin_[layerdisk]=phi;
    if (phi>phimax_[layerdisk]) phimax_[layerdisk]=phi;
    
  }
  
  void addLink(double phimin,double phimax){
    //cout << "DTC addLink "<<name_<<endl;
    DTCLink link(phimin,phimax);
    links_.push_back(link);
  }
  
  int addStub(std::pair<Stub*,L1TStub*> stub) {
    double phi=Util::phiRange(stub.second->phi());
    bool overlaplayer=((stub.second->layer()+1)%2==0);
    //cout << "layer overlaplayer : "<<stub.second->layer()+1<<" "<<overlaplayer
    //	 <<endl;
    int added=0;
    //cout << "In DTC : "<<name_<<" #links "<<links_.size()<<endl; 
    for (unsigned int i=0;i<links_.size();i++){
      if (links_[i].inRange(phi,overlaplayer)){
	added++;
	//cout << "Added stub in DTC"<<endl;
	links_[i].addStub(stub);
      }
    }
    return added;
  }

  unsigned int nLinks() const {return links_.size();}

  const DTCLink& link(unsigned int i) const {return links_[i];}

  void clean() {
    for (unsigned int i=0;i<links_.size();i++) {
      links_[i].clean();
    }
  }

  double min(unsigned int i) const {
    return phimin_[i];
  }

  double max(unsigned int i) const {
    return phimax_[i];
  }

private:

  string name_;
  std::vector<DTCLink > links_;
  std::vector<int> sectors_;

  double phimin_[11];
  double phimax_[11];


};

#endif

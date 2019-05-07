// This class holds a list of stubs that are in a given layer and DCT region
#ifndef FPGADTC_H
#define FPGADTC_H

#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGADTCLink.hh"

using namespace std;

class FPGADTC{

public:

  FPGADTC(string name=""){
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
    //cout << "FPGADTC addLink "<<name_<<endl;
    FPGADTCLink link(phimin,phimax);
    links_.push_back(link);
  }
  
  int addStub(std::pair<FPGAStub*,L1TStub*> stub) {
    double phi=FPGAUtil::phiRange(stub.second->phi());
    bool overlaplayer=((stub.second->layer()+1)%2==0);
    //cout << "layer overlaplayer : "<<stub.second->layer()+1<<" "<<overlaplayer
    //	 <<endl;
    int added=0;
    //cout << "In FPGADTC : "<<name_<<" #links "<<links_.size()<<endl; 
    for (unsigned int i=0;i<links_.size();i++){
      if (links_[i].inRange(phi,overlaplayer)){
	added++;
	//cout << "Added stub in FPGADTC"<<endl;
	links_[i].addStub(stub);
      }
    }
    return added;
  }

  unsigned int nLinks() const {return links_.size();}

  const FPGADTCLink& link(unsigned int i) const {return links_[i];}

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
  std::vector<FPGADTCLink > links_;
  std::vector<int> sectors_;

  double phimin_[11];
  double phimax_[11];


};

#endif

// This class holds a list of stubs that are in a given layer and DCT region
#ifndef FPGADTC_H
#define FPGADTC_H

#include "L1TStub.hh"
#include "FPGAStub.hh"
#include "FPGADTCLink.hh"

using namespace std;

class FPGADTC{

public:

  FPGADTC(int num=-1){
    num_=num;
  }

  void init(int num){
    num_=num;
  }
  
  void addLink(double phimin,double phimax){
    FPGADTCLink link(phimin,phimax);
    links_.push_back(link);
  }
  
  int addStub(std::pair<FPGAStub*,L1TStub*> stub) {
    double phi=stub.second->phi();
    bool overlaplayer=((stub.second->layer()+1)%2==0);
    //cout << "layer overlaplayer : "<<stub.second->layer()+1<<" "<<overlaplayer
    //	 <<endl;
    int added=0;
    //cout << "In FPGADTC : "<<num_<<" #links "<<links_.size()<<endl; 
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


private:

  int num_;
  std::vector<FPGADTCLink > links_;



};

#endif

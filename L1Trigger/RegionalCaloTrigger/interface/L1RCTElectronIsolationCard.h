#ifndef L1RCTElectronIsolationCard_h
#define L1RCTElectronIsolationCard_h

#include <vector>
#include <iostream>
#include "L1RCTRegion.h"

using std::vector;
using std::cout;
using std::endl;

class L1RCTElectronIsolationCard {

 public:

  L1RCTElectronIsolationCard(int crateNumber,
			     int cardNumber);
  ~L1RCTElectronIsolationCard();

  int crateNumber() {return crtNo;}
  int cardNumber() {return cardNo;}
  
  void fillElectronCandidates();
  void setRegion(int i, L1RCTRegion* region){
    regions.at(i) = region;
  }
  unsigned short getIsoElectrons(int i) {
    return isoElectrons.at(i);
  }
  
  unsigned short getNonIsoElectrons(int i) {
    return nonIsoElectrons.at(i);
  }
  void print();
  void printEdges(){
    regions.at(0)->printEdges();
    regions.at(1)->printEdges();
  }

 private:
  vector<unsigned short> calcElectronCandidates(L1RCTRegion *region);
  unsigned short calcMaxSum(unsigned short primaryEt,unsigned short northEt,
			    unsigned short southEt, unsigned short eastEt,
			    unsigned short westEt);

  int crtNo;
  int cardNo;

  L1RCTRegion empty;

  vector<unsigned short> isoElectrons;
  vector<unsigned short> nonIsoElectrons;
  vector<L1RCTRegion*> regions;

  L1RCTElectronIsolationCard();
};

#endif

#ifndef L1TTRACKLETS_H
#define L1TTRACKLETS_H

#include <iostream>
#include <assert.h>

#include "L1TTracklet.h"


using namespace std;


class L1TTracklets{

public:

  L1TTracklets(){
  }

  void addTracklet(const L1TTracklet& aTracklet){
    tracklets_.push_back(aTracklet);
  }

  void print() {
    for (unsigned int i=0;i<tracklets_.size();i++){
      tracklets_[i].print();
    }
  }
  
  unsigned int size() { return tracklets_.size(); }

  L1TTracklet& get(unsigned int i) { return tracklets_[i];}

  void clean(){
    tracklets_.clear();
  }

private:

  vector<L1TTracklet> tracklets_;

};


#endif


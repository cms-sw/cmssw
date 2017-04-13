//Holds the candidate matches
#ifndef FPGACANDIDATEMATCH_H
#define FPGACANDIDATEMATCH_H

#include "FPGATracklet.hh"
#include "FPGAMemoryBase.hh"
#include "FPGAStub.hh"
#include "L1TStub.hh"

using namespace std;

class FPGACandidateMatch:public FPGAMemoryBase{

public:

  FPGACandidateMatch(string name, unsigned int iSector, 
		     double phimin, double phimax):
    FPGAMemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
  }

  void addMatch(FPGATracklet* tracklet,std::pair<FPGAStub*,L1TStub*> stub) {
    std::pair<FPGATracklet*,std::pair<FPGAStub*,L1TStub*> > tmp(tracklet,stub);
    matches_.push_back(tmp);
  }

  unsigned int nMatches() const {return matches_.size();}

  FPGATracklet* getFPGATracklet(unsigned int i) const {return matches_[i].first;}
  std::pair<FPGAStub*,L1TStub*> getStub(unsigned int i) const {return matches_[i].second;}

  void clean() {
    matches_.clear();
  }


private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<FPGATracklet*,std::pair<FPGAStub*,L1TStub*> > > matches_;

};

#endif

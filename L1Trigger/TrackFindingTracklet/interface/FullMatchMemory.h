//Holds the candidate matches
#ifndef FULLMATCHMEMORY_H
#define FULLMATCHMEMORY_H

#include "Tracklet.h"
#include "MemoryBase.h"
#include "Stub.h"
#include "L1TStub.h"

using namespace std;

class FullMatchMemory:public MemoryBase{

public:

  FullMatchMemory(string name, unsigned int iSector, 
		double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
    string subname=name.substr(8,2);
    if (extended_)
      subname=name.substr(10,2);
    layer_ = 0;
    disk_  = 0;
    if (subname=="L1") layer_=1;
    if (subname=="L2") layer_=2;
    if (subname=="L3") layer_=3;
    if (subname=="L4") layer_=4;
    if (subname=="L5") layer_=5;
    if (subname=="L6") layer_=6;
    if (subname=="F1") disk_=1;
    if (subname=="F2") disk_=2;
    if (subname=="F3") disk_=3;
    if (subname=="F4") disk_=4;
    if (subname=="F5") disk_=5;
    if (subname=="D1") disk_=1;
    if (subname=="D2") disk_=2;
    if (subname=="D3") disk_=3;
    if (subname=="D4") disk_=4;
    if (subname=="D5") disk_=5;
    if (subname=="B1") disk_=-1;
    if (subname=="B2") disk_=-2;
    if (subname=="B3") disk_=-3;
    if (subname=="B4") disk_=-4;
    if (subname=="B5") disk_=-5;
    if (layer_==0&&disk_==0) {
      cout << name<<" subname = "<<subname<<" "<<layer_<<" "<<disk_<<endl;
    }
    assert((layer_!=0)||(disk_!=0));
  }

  void addMatch(Tracklet* tracklet,std::pair<Stub*,L1TStub*> stub) {
    if (!doKF) { //When using KF we allow multiple matches
      for(unsigned int i=0;i<matches_.size();i++){
	if (matches_[i].first==tracklet){ //Better match, replace
	  matches_[i].second=stub;
	  return;
	}
      }
    }
    std::pair<Tracklet*,std::pair<Stub*,L1TStub*> > tmp(tracklet,stub);
    //Check that we have the right TCID order
    if (matches_.size()>0) {
      if ( (!doKF && matches_[matches_.size()-1].first->TCID()>=tracklet->TCID()) || 
	   (doKF && matches_[matches_.size()-1].first->TCID()>tracklet->TCID()) ) {
	cout << "Wrong TCID ordering in "<<getName()<<" : "
	     <<matches_[matches_.size()-1].first->TCID()
	     <<" "<<tracklet->TCID()
	     <<" "<<matches_[matches_.size()-1].first->trackletIndex()
	     <<" "<<tracklet->trackletIndex()<<endl;
	//assert(0);
      }
    }
    matches_.push_back(tmp);
  }

  void addMatch(std::pair<Tracklet*,std::pair<Stub*,L1TStub*> > match) {
    for(unsigned int i=0;i<matches_.size();i++){
      if (matches_[i].first==match.first){
	matches_[i].second=match.second;
	return;
      }
    }
    matches_.push_back(match);
  }

  unsigned int nMatches() const {return matches_.size();}

  Tracklet* getFPGATracklet(unsigned int i) const {return matches_[i].first;}

  std::pair<Tracklet*,std::pair<Stub*,L1TStub*> > getMatch(unsigned int i) const {return matches_[i];}

  void clean() {
    matches_.clear();
  }

  void writeMC(bool first) {

    std::string fname="../data/MemPrints/Matches/FullMatches_";
    fname+=getName();
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_=0;
      event_=1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    for (unsigned int j=0;j<matches_.size();j++){
      string match= (layer_>0)? matches_[j].first->fullmatchstr(layer_)
	: matches_[j].first->fullmatchdiskstr(disk_);
      out_ << "0x";
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ << " "<< match <<" "<<hexFormat(match)<<endl;
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

    
  }

  int layer() const { return layer_;}
  int disk() const { return disk_;}


private:

  double phimin_;
  double phimax_;
  std::vector<std::pair<Tracklet*,std::pair<Stub*,L1TStub*> > > matches_;

  int layer_;
  int disk_;

};

#endif

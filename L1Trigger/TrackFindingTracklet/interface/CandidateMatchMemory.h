//Holds the candidate matches
#ifndef CANDIDATEMATCHMEMORY_H
#define CANDIDATEMATCHMEMORY_H

#include "Tracklet.h"
#include "MemoryBase.h"
#include "Stub.h"
#include "L1TStub.h"

using namespace std;

class CandidateMatchMemory:public MemoryBase{

public:

  CandidateMatchMemory(string name, unsigned int iSector, 
		     double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
    string subname=name.substr(3,2);
    layer_ = 0;
    disk_ = 0;
    if (subname=="L1") layer_=1;
    if (subname=="L2") layer_=2;
    if (subname=="L3") layer_=3;
    if (subname=="L4") layer_=4;
    if (subname=="L5") layer_=5;
    if (subname=="L6") layer_=6;

    if (subname=="D1") disk_=1;
    if (subname=="D2") disk_=2;
    if (subname=="D3") disk_=3;
    if (subname=="D4") disk_=4;
    if (subname=="D5") disk_=5;

    if (layer_==0&&disk_==0) {
      cout << name<<" subname = "<<subname<<" "<<layer_<<" "<<disk_<<endl;
    }   
    assert((layer_!=0)||(disk_!=0)); 
  }

  void addMatch(std::pair<Tracklet*,int> tracklet,std::pair<Stub*,L1TStub*> stub) {
    std::pair<std::pair<Tracklet*,int>,std::pair<Stub*,L1TStub*> > tmp(tracklet,stub);

    //Check for consistency
    for(unsigned int i=0;i<matches_.size();i++){
      if (tracklet.first->TCID()<matches_[i].first.first->TCID()) {
	cout << "In "<<getName()<<" adding tracklet "<<tracklet.first<<" with lower TCID : "
	     <<tracklet.first->TCID()<<" than earlier TCID "<<matches_[i].first.first->TCID()<<endl;
	assert(0);
      }
    }
    
    matches_.push_back(tmp);    
  }

  unsigned int nMatches() const {return matches_.size();}

  Tracklet* getFPGATracklet(unsigned int i) const {return matches_[i].first.first;}
  std::pair<Stub*,L1TStub*> getStub(unsigned int i) const {return matches_[i].second;}
  std::pair<std::pair<Tracklet*,int>,std::pair<Stub*,L1TStub*> >
	getMatch(unsigned int i) const {return matches_[i];}

  void clean() {
    matches_.clear();
  }

  void writeCM(bool first) {

    std::string fname="../data/MemPrints/Matches/CandidateMatches_";
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
      string stubid = matches_[j].second.first->stubindex().str(); // stub ID
      int projindex= (layer_>0)? matches_[j].first.second
        : matches_[j].first.second; // Allproj index
      FPGAWord tmp;
      if (projindex>=(1<<7)) {
	projindex=(1<<7)-1;
      }
      tmp.set(projindex,7,true,__LINE__,__FILE__);
      out_ << "0x";
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ << " "<< tmp.str() <<"|"<< stubid << " " << hexFormat(tmp.str()+stubid)<<endl;
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
  std::vector<std::pair<std::pair<Tracklet*, int>,std::pair<Stub*,L1TStub*> > > matches_;

  int layer_;
  int disk_;

};

#endif

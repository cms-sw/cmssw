#ifndef ALLPROJECTIONSMEMORY_H
#define ALLPROJECTIONSMEMORY_H

#include "Tracklet.h"
#include "MemoryBase.h"

using namespace std;

class AllProjectionsMemory:public MemoryBase{

public:

  AllProjectionsMemory(string name, unsigned int iSector, 
		     double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
    string subname=name.substr(3,2);
    if (subname[0]=='_') subname=name.substr(9,2);
    
    layer_ = 0; 
    disk_  = 0;
 
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
    if (subname=="F1") disk_=1;
    if (subname=="F2") disk_=2;
    if (subname=="F3") disk_=3;
    if (subname=="F4") disk_=4;
    if (subname=="F5") disk_=5;
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

  void addTracklet(Tracklet* tracklet) {
    tracklets_.push_back(tracklet);
  }

  unsigned int nTracklets() const {return tracklets_.size();}

  Tracklet* getFPGATracklet(unsigned int i) const {return tracklets_[i];}

  void clean() {
    tracklets_.clear();
  }

void writeAP(bool first) {

  std::string fname="../MemPrints/TrackletProjections/AllProj_";
  fname+=getName();
  fname+="_";
  ostringstream oss;
  oss << iSector_+1; 
  if (iSector_+1<10) fname+="0";
  fname+=oss.str();
  fname+=".dat";
  if (first){
    bx_=0;
    event_=1;
    out_.open(fname.c_str());
  }
  else
    out_.open(fname.c_str(),std::ofstream::app);

  out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j=0;j<tracklets_.size();j++){
    string proj= (layer_>0)? tracklets_[j]->trackletprojstrlayer(layer_)
      : tracklets_[j]->trackletprojstrdisk(disk_);
	  out_ << "0x";
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ << " "<< proj << "  "<<hexFormat(proj)<<endl;
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
  std::vector<Tracklet*> tracklets_;

  int layer_;
  int disk_;

};

#endif

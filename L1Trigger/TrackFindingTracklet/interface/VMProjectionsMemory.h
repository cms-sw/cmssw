//Holds the VM projections
#ifndef VMPROJECTIONSMEMORY_H
#define VMPROJECTIONSMEMORY_H

#include "Tracklet.h"
#include "MemoryBase.h"

using namespace std;

class VMProjectionsMemory:public MemoryBase{

public:

  VMProjectionsMemory(string name, unsigned int iSector, 
		    double phimin, double phimax):
    MemoryBase(name,iSector){
    phimin_=phimin;
    phimax_=phimax;
    string subname=name.substr(7,2);
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
    if (layer_==0&&disk_==0) {
      cout << name<<" subname = "<<subname<<" "<<layer_<<" "<<disk_<<endl;
    }
    assert((layer_!=0)||(disk_!=0));
  }

  void addTracklet(Tracklet* tracklet,unsigned int allprojindex) {
    std::pair<Tracklet*,unsigned int> tmp(tracklet,allprojindex);
    //Check that order of TCID is correct
    if (tracklets_.size()>0) {
      assert(tracklets_[tracklets_.size()-1].first->TCID()<=tracklet->TCID());
    }
    tracklets_.push_back(tmp);
  }

  unsigned int nTracklets() const {return tracklets_.size();}

  Tracklet* getFPGATracklet(unsigned int i) const {return tracklets_[i].first;}
  int getAllProjIndex(unsigned int i) const {return tracklets_[i].second;}

  void writeVMPROJ(bool first) {

    //cout << "In writeTPROJ "<<tracklets_.size()<<"\t"<<name_<<" "<<layer_<<" "<<disk_<<endl;

    std::string fname="../data/MemPrints/VMProjections/VMProjections_";
    fname+=getName();
    //get rid of duplicates
    int len = fname.size();
    if(fname[len-2]=='n'&& fname[len-1]>'1'&&fname[len-1]<='9') return;
    //
    fname+="_";
    ostringstream oss;
    oss << iSector_+1;
    if (iSector_+1<10) fname+="0";
    fname+=oss.str();
    fname+=".dat";
    if (first) {
      bx_ = 0;
      event_ = 1;
      out_.open(fname.c_str());
    }
    else
      out_.open(fname.c_str(),std::ofstream::app);

    out_ << "BX = "<<(bitset<3>)bx_ << " Event : " << event_ << endl;

    for (unsigned int j=0;j<tracklets_.size();j++){
      string vmproj=(layer_>0)? tracklets_[j].first->vmstrlayer(layer_,tracklets_[j].second)
	: tracklets_[j].first->vmstrdisk(disk_,tracklets_[j].second);
      out_ << "0x";
      if (j<16) out_ <<"0";
      out_ << hex << j << dec ;
      out_ <<" "<<vmproj<<" "<<hexFormat(vmproj)<<endl;
    }
    out_.close();

    bx_++;
    event_++;
    if (bx_>7) bx_=0;

  }

  void clean() {
    tracklets_.clear();
  }

  int layer() const { return layer_;}
  int disk() const { return disk_;}

private:

  double phimin_;
  double phimax_;
  int layer_;
  int disk_;
  std::vector<std::pair<Tracklet*,unsigned int> > tracklets_;

};

#endif

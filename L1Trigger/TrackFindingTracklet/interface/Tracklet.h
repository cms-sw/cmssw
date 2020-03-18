#ifndef TRACKLET_H
#define TRACKLET_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <set>
#include "L1TStub.h"
#include "Stub.h"
#include "FPGAWord.h"
#include "Track.h"
#include "TrackPars.h"
#include "LayerProjection.h"
#include "DiskProjection.h"
#include "LayerResidual.h"
#include "DiskResidual.h"
#include "Util.h"

using namespace std;

class Tracklet{

public:

  Tracklet(L1TStub* innerStub, L1TStub* middleStub, L1TStub* outerStub,
	   Stub* innerFPGAStub, Stub* middleFPGAStub, Stub* outerFPGAStub,
	   double rinv, double phi0, double d0, double z0, double t,
	   double rinvapprox, double phi0approx, double d0approx,
	   double z0approx, double tapprox,
	   int irinv, int iphi0, int id0,
	   int iz0, int it,
	   LayerProjection layerprojs[4],
	   DiskProjection diskprojs[4],
	   bool disk, bool overlap=false){

    
    //cout << "New Tracklet"<<endl;

    overlap_=overlap;
    disk_=disk;
    assert(!(disk&&overlap));
    barrel_=(!disk)&&(!overlap);
    triplet_ = false;

    trackletIndex_=-1;
    TCIndex_=-1;

    assert(disk_||barrel_||overlap_);

    if (barrel_ && middleStub==NULL) assert(innerStub->layer()<6);
  
    innerStub_=innerStub;
    middleStub_=middleStub;
    outerStub_=outerStub;
    innerFPGAStub_=innerFPGAStub;
    middleFPGAStub_=middleFPGAStub;
    outerFPGAStub_=outerFPGAStub;

    trackpars_.init(rinv,phi0,d0,t,z0);

    trackparsapprox_.init(rinvapprox,phi0approx,d0approx,tapprox,z0approx);

    fpgapars_.rinv().set(irinv,nbitsrinv,false,__LINE__,__FILE__); 
    fpgapars_.phi0().set(iphi0,nbitsphi0,false,__LINE__,__FILE__); 
    fpgapars_.d0().set(id0,nbitsd0,false,__LINE__,__FILE__); 
    fpgapars_.z0().set(iz0,nbitsz0,false,__LINE__,__FILE__);
    fpgapars_.t().set(it,nbitst,false,__LINE__,__FILE__);       

    fpgatrack_ = NULL;
    
    if (innerStub_) assert(innerStub_->layer()<6||innerStub_->disk()<5);
    if (middleStub_) assert(middleStub_->layer()<6||middleStub_->disk()<5);
    if (outerStub_) assert(outerStub_->layer()<6||outerStub_->disk()<5);

    seedIndex_=calcSeedIndex();

    triplet_=(seedIndex_>=8);

    //fill projection layers
    for(unsigned int i=0;i<4;i++) {
      projlayer_[i]=projlayers[seedIndex_][i];
    }

    //fill projection disks
    for(unsigned int i=0;i<5;i++) {
      projdisk_[i]=projdisks[seedIndex_][i];
    }

    
    //Handle projections to the layers
    for (int i=0;i<4;i++) {
      
      if (projlayer_[i]==0) continue;
      if (!layerprojs[i].valid()) continue;
      
      layerproj_[projlayer_[i]-1]=layerprojs[i];
      
    }
    //Now handle projections to the disks
    for (int i=0;i<5;i++) {
      
      if (projdisk_[i]==0) continue;
      if (!diskprojs[i].valid()) continue;
	
      diskproj_[projdisk_[i]-1]=diskprojs[i];
      
    }
    
    
    ichisqfit_.set(-1,8,false);

    
  }
  

  ~Tracklet() {

    delete fpgatrack_;

  }



  //Find tp corresponding to seed.
  //Will require 'tight match' such that tp is part of
  //each of the four clustes
  //returns 0 if no tp matches
  int tpseed() {

    set<int> tpset;

    set<int> tpsetstubinner;
    set<int> tpsetstubouter;

    vector<int> tps=innerStub_->tps();
    for(unsigned int i=0;i<tps.size();i++){
      if (tps[i]!=0) {
	tpsetstubinner.insert(tps[i]);
	tpset.insert(abs(tps[i]));
      }
    }
    
    tps=outerStub_->tps();
    for(unsigned int i=0;i<tps.size();i++){
      if (tps[i]!=0) {
	tpsetstubouter.insert(tps[i]);
	tpset.insert(abs(tps[i]));
      }
    }

    for(auto seti=tpset.begin();seti!=tpset.end();seti++){
      int tp=*seti;
      if (tpsetstubinner.find(tp)!=tpsetstubinner.end()&&
	  tpsetstubinner.find(-tp)!=tpsetstubinner.end()&&
	  tpsetstubouter.find(tp)!=tpsetstubouter.end()&&
	  tpsetstubouter.find(-tp)!=tpsetstubouter.end()) {
	return tp;
      }
    }
    return 0;
  }
  

  bool stubtruthmatch(L1TStub* stub){

    set<int> tpset;

    set<int> tpsetstub;
    set<int> tpsetstubinner;
    set<int> tpsetstubouter;

    vector<int> tps=stub->tps();
    for(unsigned int i=0;i<tps.size();i++){
      if (tps[i]!=0) {
	tpsetstub.insert(tps[i]);
	tpset.insert(abs(tps[i]));
      }
    }
    tps=innerStub_->tps();
    for(unsigned int i=0;i<tps.size();i++){
      if (tps[i]!=0) {
	tpsetstubinner.insert(tps[i]);
	tpset.insert(abs(tps[i]));
      }
    }
    tps=outerStub_->tps();
    for(unsigned int i=0;i<tps.size();i++){
      if (tps[i]!=0) {
	tpsetstubouter.insert(tps[i]);
	tpset.insert(abs(tps[i]));
      }
    }

    for(auto seti=tpset.begin();seti!=tpset.end();seti++){
      int tp=*seti;
      if (tpsetstub.find(tp)!=tpsetstub.end()&&
	  tpsetstub.find(-tp)!=tpsetstub.end()&&
	  tpsetstubinner.find(tp)!=tpsetstubinner.end()&&
	  tpsetstubinner.find(-tp)!=tpsetstubinner.end()&&
	  tpsetstubouter.find(tp)!=tpsetstubouter.end()&&
	  tpsetstubouter.find(-tp)!=tpsetstubouter.end()) {
	return true;
      }
      
    }

    return false;
    
  }

  L1TStub* innerStub() {return innerStub_;}
  Stub* innerFPGAStub() {return innerFPGAStub_;}

  L1TStub* middleStub() {return middleStub_;}
  Stub* middleFPGAStub() {return middleFPGAStub_;}

  L1TStub* outerStub() {return outerStub_;}
  Stub* outerFPGAStub() {return outerFPGAStub_;}

  std::string addressstr() {
    std::ostringstream oss;
    oss << innerFPGAStub_->phiregionaddressstr()<<"|";
    if (middleFPGAStub_) {
      oss << middleFPGAStub_->phiregionaddressstr()<<"|";
    }
    oss << outerFPGAStub_->phiregionaddressstr();
    
    return oss.str();
    
  }
  
  
  //Tracklet parameters print out
  std::string trackletparstr() {
    std::ostringstream oss;
    if(writeoutReal){
      oss << fpgapars_.rinv().value()*krinvpars<<" "
	  << fpgapars_.phi0().value()*kphi0pars<<" "
	  << fpgapars_.d0().value()*kd0pars<<" "
	  << fpgapars_.z0().value()*kz<<" "
	  << fpgapars_.t().value()*ktpars;
    }
    
    //Binary Print out
    if(!writeoutReal){
      oss << innerFPGAStub_->stubindex().str()<<"|";
      if (middleFPGAStub_) {
        oss << middleFPGAStub_->stubindex().str()<<"|";
      }
      oss << outerFPGAStub_->stubindex().str()<<"|"
	  << fpgapars_.rinv().str()<<"|"
	  << fpgapars_.phi0().str()<<"|"
	  << fpgapars_.d0().str()<<"|"
	  << fpgapars_.z0().str()<<"|"
	  << fpgapars_.t().str();
    }

    return oss.str();
  }
  
  std::string vmstrlayer(int layer, unsigned int allstubindex) {
    std::ostringstream oss;
    FPGAWord index;
    if (allstubindex>=(1<<7)) {
      cout << "Warning projection number too large!"<<endl;	
      index.set((1<<7)-1,7,true,__LINE__,__FILE__);
    } else {
      index.set(allstubindex,7,true,__LINE__,__FILE__);
    }

    // This is a shortcut.
    //int irinvvm=16+(fpgarinv().value()>>(fpgarinv().nbits()-5));
    // rinv is not directly available in the TrackletProjection.
    // can be inferred from phi derivative: rinv = - phider * 2
    int tmp_irinv = layerproj_[layer-1].fpgaphiprojder().value()*(-2);
    int nbits_irinv = layerproj_[layer-1].fpgaphiprojder().nbits()+1;
    
    // irinv in VMProjection: 
    // top 5 bits of rinv and shifted to be positive
    int irinvvm = 16+(tmp_irinv>>(nbits_irinv-5));

    assert(irinvvm>=0);
    assert(irinvvm<32);
    FPGAWord tmp;
    tmp.set(irinvvm,5,true,__LINE__,__FILE__);
    oss << index.str()
	<<"|"<< layerproj_[layer-1].fpgazbin1projvm().str() 
        <<"|"<< layerproj_[layer-1].fpgazbin2projvm().str()
	<<"|"<< layerproj_[layer-1].fpgafinezvm().str()
	<<"|"<< tmp.str()<<"|"<<PSseed();
    return oss.str();

  }
  
  std::string vmstrdisk(int disk, unsigned int allstubindex) {
    std::ostringstream oss;
    FPGAWord index;
    if (allstubindex>=(1<<7)) {
      cout << "Warning projection number too large!"<<endl;	
      index.set((1<<7)-1,7,true,__LINE__,__FILE__);
    } else {
      index.set(allstubindex,7,true,__LINE__,__FILE__);
    } 
    oss << index.str()
	<<"|"<<diskproj_[disk-1].fpgarbin1projvm().str()
	<<"|"<<diskproj_[disk-1].fpgarbin2projvm().str()
	<<"|"<<diskproj_[disk-1].fpgafinervm().str()
	<<"|"<< diskproj_[disk-1].getBendIndex().str();
    return oss.str();

  }

  
  std::string trackletprojstr(int layer) const {
    assert(layer>=1&&layer<=6);
    std::ostringstream oss;
    FPGAWord tmp;
    if (trackletIndex_<0||trackletIndex_>127) {
      cout << "trackletIndex_ = "<<trackletIndex_<<endl;
      assert(0);
    }
    tmp.set(trackletIndex_,7,true,__LINE__,__FILE__);
    FPGAWord tcid;
    tcid.set(TCIndex_,7,true,__LINE__,__FILE__);

    oss << tcid.str()<<"|"
	<< tmp.str()<<"|"
        << layerproj_[layer-1].fpgaphiproj().str()<<"|"
	<< layerproj_[layer-1].fpgazproj().str()<<"|"
	<< layerproj_[layer-1].fpgaphiprojder().str()<<"|"
	<< layerproj_[layer-1].fpgazprojder().str();

    return oss.str();
    
  }
  std::string trackletprojstrD(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    std::ostringstream oss;
    FPGAWord tmp;
    if (trackletIndex_<0||trackletIndex_>127) {
      cout << "trackletIndex_ = "<<trackletIndex_<<endl;
      assert(0);
    }
    tmp.set(trackletIndex_,7,true,__LINE__,__FILE__);
    FPGAWord tcid;
    tcid.set(TCIndex_,7,true,__LINE__,__FILE__);
    oss << tcid.str()<<"|" 
        << tmp.str()<<"|"
	<< diskproj_[abs(disk)-1].fpgaphiproj().str()<<"|"
	<< diskproj_[abs(disk)-1].fpgarproj().str()<<"|"
	<< diskproj_[abs(disk)-1].fpgaphiprojder().str()<<"|"
	<< diskproj_[abs(disk)-1].fpgarprojder().str();

    return oss.str();

  }

  std::string trackletprojstrlayer(int layer) const {
    return trackletprojstr(layer);
  }
  std::string trackletprojstrdisk(int disk) const {
    std::ostringstream oss;
    return trackletprojstrD(disk);

  }

  bool validProj(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].valid();
  }

  FPGAWord fpgaphiprojder(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgaphiprojder();
  }

  FPGAWord fpgazproj(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgazproj();
  }

  FPGAWord fpgaphiproj(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgaphiproj();
  }

  FPGAWord fpgazprojder(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgazprojder();
  }

  int zbin1projvm(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgazbin1projvm().value();
  }

  int zbin2projvm(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgazbin2projvm().value();
  }

  int finezvm(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgafinezvm().value();
  }

  int rbin1projvm(int disk) const {
    assert(disk>=1&&disk<=5);
    return diskproj_[disk-1].fpgarbin1projvm().value();
  }

  int rbin2projvm(int disk) const {
    assert(disk>=1&&disk<=5);
    return diskproj_[disk-1].fpgarbin2projvm().value();
  }

  int finervm(int disk) const {
    assert(disk>=1&&disk<=5);
    return diskproj_[disk-1].fpgafinervm().value();
  }

  int phiprojvm(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgaphiprojvm().value();
  }

  int zprojvm(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].fpgazprojvm().value();
  }


  
  double phiproj(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].phiproj();
  }

  double phiprojder(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].phiprojder();
  }

  double zproj(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].zproj();
  }

  double zprojder(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].zprojder();
  }



  
  double zprojapprox(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].zprojapprox();
  }

  double zprojderapprox(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].zprojderapprox();
  }

  double phiprojapprox(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].phiprojapprox();
  }

  double phiprojderapprox(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].phiprojderapprox();
  }

  

  double rproj(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerproj_[layer-1].rproj();
  }


  double rstub(int layer) {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].rstub();
  }





  //Disks residuals

  bool validProjDisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].valid();
  }


  FPGAWord fpgaphiresiddisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].fpgaphiresid();
  }

  FPGAWord fpgarresiddisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].fpgarresid();
  }

  double phiresiddisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].phiresid();
  }

  double rresiddisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].rresid();
  }

  double phiresidapproxdisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].phiresidapprox();
  }

  double rresidapproxdisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].rresidapprox();
  }


  double zstubdisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].zstub();
  }

  void setBendIndex(int bendIndex,int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    diskproj_[abs(disk)-1].setBendIndex(bendIndex);
  }

  FPGAWord getBendIndex(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].getBendIndex();
  }
  

  double alphadisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].alpha();
  }

  FPGAWord ialphadisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].ialpha();
  }

  

  FPGAWord fpgaphiprojdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].fpgaphiproj();
  }

  FPGAWord fpgaphiprojderdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].fpgaphiprojder();
  }

  FPGAWord fpgarprojdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].fpgarproj();
  }
  
  FPGAWord fpgarprojderdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].fpgarprojder();
  }

  

  double phiprojapproxdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].phiprojapprox();
  }

  double phiprojderapproxdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].phiprojderapprox();
  }
  
  double rprojapproxdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].rprojapprox();
  }

  double rprojderapproxdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].rprojderapprox();
  }



  double phiprojdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].phiproj();
  }

  double phiprojderdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].phiprojder();
  }
  
  double rprojdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].rproj();
  }
  
  double rprojderdisk(int disk) const {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskproj_[abs(disk)-1].rprojder();
  }


  bool matchdisk(int disk) {
    assert(abs(disk)>=1&&abs(disk)<=5);
    return diskresid_[abs(disk)-1].valid();
  }

  void addMatch(int layer, int ideltaphi, int ideltaz, 
		double dphi, double dz, 
		double dphiapprox, double dzapprox, 
		int stubid,double rstub,
		std::pair<Stub*,L1TStub*> stubptrs){
    
    assert(layer>=1&&layer<=6);
    
    layerresid_[layer-1].init(layer,ideltaphi,ideltaz,stubid,dphi,dz,dphiapprox,dzapprox,rstub,stubptrs);
    
  }
  
  
  
  void addMatchDisk(int disk, int ideltaphi, int ideltar, 
		    double dphi, double dr, 
		    double dphiapprox, double drapprox, double alpha,
		    int stubid,double zstub,
		    std::pair<Stub*,L1TStub*> stubptrs){
    
    assert(abs(disk)>=1&&abs(disk)<=5);
    
    diskresid_[abs(disk)-1].init(disk,ideltaphi,ideltar,stubid,dphi,dr,dphiapprox,drapprox,zstub,alpha,stubptrs.first->alphanew(),stubptrs);
    
  }

  int nMatches() {

    int nmatches=0;

    for (int i=0;i<6;i++) {
      if (layerresid_[i].valid()) {
	nmatches++;
      }
    }
    
    return nmatches;
    
  }

  int nMatchesDisk() {

    int nmatches=0;
    
    for (int i=0;i<5;i++) {
      if (diskresid_[i].valid()) {
	nmatches++;
      }
    }
    return nmatches;

  }



  bool match(int layer) {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].valid();
  }

  
  std::string fullmatchstr(int layer) {
    assert(layer>=1&&layer<=6);
    std::ostringstream oss;

    FPGAWord tmp;
    if (trackletIndex_<0||trackletIndex_>127) {
      cout << "trackletIndex_ = "<<trackletIndex_<<endl;
      assert(0);
    }
    tmp.set(trackletIndex_,7,true,__LINE__,__FILE__);
    FPGAWord tcid;
    tcid.set(TCIndex_,7,true,__LINE__,__FILE__);
    oss << tcid.str()<<"|"
        << tmp.str()<<"|"
	<< layerresid_[layer-1].fpgastubid().str()<<"|"
	<< layerresid_[layer-1].fpgaphiresid().str()<<"|"
	<< layerresid_[layer-1].fpgazresid().str();
    
    return oss.str();
    
  }

  std::string fullmatchdiskstr(int disk) {
    assert(disk>=1&&disk<=5);
    std::ostringstream oss;

    FPGAWord tmp;
    if (trackletIndex_<0||trackletIndex_>127) {
      cout << "trackletIndex_ = "<<trackletIndex_<<endl;
      assert(0);
    }
    tmp.set(trackletIndex_,7,true,__LINE__,__FILE__);
    FPGAWord tcid;
    tcid.set(TCIndex_,7,true,__LINE__,__FILE__);
    oss << tcid.str()<<"|"
        << tmp.str()<<"|"
	<< diskresid_[disk-1].fpgastubid().str()<<"|"
	<< diskresid_[disk-1].fpgaphiresid().str()<<"|"
	<< diskresid_[disk-1].fpgarresid().str();
    
    return oss.str();
    
  }


  bool validResid(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].valid();
  }

  
  std::pair<Stub*,L1TStub*> stubptrs(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].stubptrs();
  }

  
  double phiresid(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].phiresid();
  }

  double phiresidapprox(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].phiresidapprox();
  }

  double zresid(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].zresid();
  }

  double zresidapprox(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].zresidapprox();
  }





  FPGAWord fpgaphiresid(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].fpgaphiresid();
  }

  FPGAWord fpgazresid(int layer) const {
    assert(layer>=1&&layer<=6);
    return layerresid_[layer-1].fpgazresid();
  }


  std::vector<L1TStub*> getL1Stubs() {

    std::vector<L1TStub*> tmp;

    if (innerStub_) tmp.push_back(innerStub_);
    if (middleStub_) tmp.push_back(middleStub_);
    if (outerStub_) tmp.push_back(outerStub_);

    for (unsigned int i=0;i<6;i++) {
      if (layerresid_[i].valid()) {
	tmp.push_back(layerresid_[i].stubptrs().second);
      }
    }

    for (unsigned int i=0;i<5;i++) {
      if (diskresid_[i].valid()) tmp.push_back(diskresid_[i].stubptrs().second);
    }

    return tmp;

  }

  std::map<int, int> getStubIDs() {


    std::map<int, int> stubIDs;
    
    // For future reference, *resid_[i] uses i as the absolute stub index. (0-5 for barrel, 0-4 for disk)
    // On the other hand, proj*_[i] uses i almost like *resid_[i], except the *seeding* layer indices are removed entirely.
    // E.g. An L3L4 track has 0=L1, 1=L2, 2=L4, 3=L5 for the barrels (for proj*_[i])

    if (innerFPGAStub_) assert(innerFPGAStub_->stubindex().nbits()==7);
    if (innerFPGAStub_) assert(innerFPGAStub_->phiregion().nbits()==3);
    if (middleFPGAStub_) assert(middleFPGAStub_->stubindex().nbits()==7);
    if (middleFPGAStub_) assert(middleFPGAStub_->phiregion().nbits()==3);
    if (outerFPGAStub_) assert(outerFPGAStub_->stubindex().nbits()==7);
    if (outerFPGAStub_) assert(outerFPGAStub_->phiregion().nbits()==3);
  
    if(barrel_) {
      for(int i=0; i<6; i++) {
	
        //check barrel
	if (layerresid_[i].valid()) {
	  // two extra bits to indicate if the matched stub is local or from neighbor
	  int location = 1;  // local
	  location<<=layerresid_[i].fpgastubid().nbits();
	  
	  stubIDs[1+i] = layerresid_[i].fpgastubid().value()+location;
        }             
	
	//check disk
	if (i>=5) continue; //i=[0..4] for disks
	if(diskresid_[i].valid()) {
	  if(i==3 && layerresid_[0].valid() && innerFPGAStub_->layer().value()==1) continue; // Don't add D4 if track has L1 stub
	  // two extra bits to indicate if the matched stub is local or from neighbor
	  int location = 1;  // local
	  location<<=diskresid_[i].fpgastubid().nbits();
	  
	  if(itfit().value() < 0) {
	    stubIDs[-11-i] = diskresid_[i].fpgastubid().value()+location;
	  } else {
	    stubIDs[11+i] = diskresid_[i].fpgastubid().value()+location;
	  }  
	}                     
      }
      
      
      //get stubs making up tracklet
      //printf(" inner %i  outer %i layers \n",innerFPGAStub_.layer().value(),outerFPGAStub_.layer().value());
      if (innerFPGAStub_) stubIDs[innerFPGAStub_->layer().value()+1] = ((innerFPGAStub_->phiregion().value())<<7)+innerFPGAStub_->stubindex().value()+(1<<10);
      if (middleFPGAStub_) stubIDs[middleFPGAStub_->layer().value()+1] = ((middleFPGAStub_->phiregion().value())<<7)+middleFPGAStub_->stubindex().value()+(1<<10);
      if (outerFPGAStub_) stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->phiregion().value())<<7)+outerFPGAStub_->stubindex().value()+(1<<10);
      
    } else if (disk_) {
      for(int i=0; i<5; i++) {
	
	//check barrel
	if(layerresid_[i].valid()) {
	  // two extra bits to indicate if the matched stub is local or from neighbor
	  int location = 1;  // local
	  location<<=layerresid_[i].fpgastubid().nbits();
	  
	  stubIDs[1+i] = layerresid_[i].fpgastubid().value()+location;
        }
	
	//check disks
	if(i==4 && layerresid_[1].valid()) continue; // Don't add D5 if track has L2 stub
	if(diskresid_[i].valid()) {
	  // two extra bits to indicate if the matched stub is local or from neighbor
	  int location = 1;  // local
	  location<<=diskresid_[i].fpgastubid().nbits();
	  
	  if(innerStub_->disk() < 0) {
	    stubIDs[-11-i] = diskresid_[i].fpgastubid().value()+location;
	  } else {
	    stubIDs[11+i] = diskresid_[i].fpgastubid().value()+location;
	  }
	}         
      }
      
      //get stubs making up tracklet
      //printf(" inner %i  outer %i disks \n",innerFPGAStub_.disk().value(),outerFPGAStub_.disk().value());
      if(innerFPGAStub_->disk().value() < 0) { //negative side runs 6-10
	if (innerFPGAStub_) stubIDs[innerFPGAStub_->disk().value()-10] = ((innerFPGAStub_->phiregion().value())<<7)+innerFPGAStub_->stubindex().value()+(1<<10);
	if (middleFPGAStub_) stubIDs[middleFPGAStub_->disk().value()-10] = ((middleFPGAStub_->phiregion().value())<<7)+middleFPGAStub_->stubindex().value()+(1<<10);
	if (outerFPGAStub_) stubIDs[outerFPGAStub_->disk().value()-10] = ((outerFPGAStub_->phiregion().value())<<7)+outerFPGAStub_->stubindex().value()+(1<<10);
      } else { // positive side runs 11-15]
	if (innerFPGAStub_) stubIDs[innerFPGAStub_->disk().value()+10] = ((innerFPGAStub_->phiregion().value())<<7)+innerFPGAStub_->stubindex().value()+(1<<10);
	if (middleFPGAStub_) stubIDs[middleFPGAStub_->disk().value()+10] = ((middleFPGAStub_->phiregion().value())<<7)+middleFPGAStub_->stubindex().value()+(1<<10);
	if (outerFPGAStub_) stubIDs[outerFPGAStub_->disk().value()+10] = ((outerFPGAStub_->phiregion().value())<<7)+outerFPGAStub_->stubindex().value()+(1<<10);
      }     
      
    } else if (overlap_) {
      
      for(int i=0; i<5; i++) {
	
	//check barrel
	if(layerresid_[i].valid()) {
	  // two extra bits to indicate if the matched stub is local or from neighbor
	  int location = 1;  // local
	  location<<=layerresid_[i].fpgastubid().nbits();
	  
	  stubIDs[1+i] = layerresid_[i].fpgastubid().value()+location;
	}
	
	//check disks
	if(diskresid_[i].valid()) {
	  // two extra bits to indicate if the matched stub is local or from neighbor
	  int location = 1;  // local
	  location<<=diskresid_[i].fpgastubid().nbits();
	  
	  if(innerStub_->disk() < 0) { // if negative overlap
            if(innerFPGAStub_->layer().value()!=2 || !layerresid_[0].valid() || i!=3 ) { // Don't add D4 if this is an L3L2 track with an L1 stub
	      stubIDs[-11-i] = diskresid_[i].fpgastubid().value()+location;
            }
	  } else {
            if(innerFPGAStub_->layer().value()!=2 || !layerresid_[0].valid() || i!=3 ) {
	      stubIDs[11+i] = diskresid_[i].fpgastubid().value()+location;
            }
	  }
	}         
      }
      
      //get stubs making up tracklet

      if(innerFPGAStub_->layer().value()==2) { // L3L2 track
	if (innerFPGAStub_) stubIDs[innerFPGAStub_->layer().value()+1] = ((innerFPGAStub_->phiregion().value())<<7)+innerFPGAStub_->stubindex().value()+(1<<10);
	if (middleFPGAStub_) stubIDs[middleFPGAStub_->layer().value()+1] = ((middleFPGAStub_->phiregion().value())<<7)+middleFPGAStub_->stubindex().value()+(1<<10);
	if (outerFPGAStub_) stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->phiregion().value())<<7)+outerFPGAStub_->stubindex().value()+(1<<10);
      } else if(innerFPGAStub_->disk().value() < 0) { //negative side runs -11 - -15
	if (innerFPGAStub_) stubIDs[innerFPGAStub_->disk().value()-10] = ((innerFPGAStub_->phiregion().value())<<7)+innerFPGAStub_->stubindex().value()+(1<<10);
	if (middleFPGAStub_) stubIDs[middleFPGAStub_->layer().value()+1] = ((middleFPGAStub_->phiregion().value())<<7)+middleFPGAStub_->stubindex().value()+(1<<10);
	if (outerFPGAStub_) stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->phiregion().value())<<7)+outerFPGAStub_->stubindex().value()+(1<<10);
      } else { // positive side runs 11-15]
	if (innerFPGAStub_) stubIDs[innerFPGAStub_->disk().value()+10] = ((innerFPGAStub_->phiregion().value())<<7)+innerFPGAStub_->stubindex().value()+(1<<10);
	if (middleFPGAStub_) stubIDs[middleFPGAStub_->layer().value()+1] = ((middleFPGAStub_->phiregion().value())<<7)+middleFPGAStub_->stubindex().value()+(1<<10);
	if (outerFPGAStub_) stubIDs[outerFPGAStub_->layer().value()+1] = ((outerFPGAStub_->phiregion().value())<<7)+outerFPGAStub_->stubindex().value()+(1<<10);
      }     
      
    }
    
    
    return stubIDs;
  }
  

  double rinv() const { return trackpars_.rinv(); }
  double phi0() const { return trackpars_.phi0(); }
  double d0() const { return trackpars_.d0(); }
  double t() const { return trackpars_.t(); }
  double z0() const { return trackpars_.z0(); }

  double rinvapprox() const { return trackparsapprox_.rinv(); }
  double phi0approx() const { return trackparsapprox_.phi0(); }
  double d0approx() const { return trackparsapprox_.d0(); }
  double tapprox() const { return trackparsapprox_.t(); }
  double z0approx() const { return trackparsapprox_.z0(); }


  FPGAWord fpgarinv() const { return fpgapars_.rinv(); }
  FPGAWord fpgaphi0() const { return fpgapars_.phi0(); }
  FPGAWord fpgad0() const { return fpgapars_.d0(); }
  FPGAWord fpgat() const { return fpgapars_.t(); }
  FPGAWord fpgaz0() const { return fpgapars_.z0(); }

  double rinvfit() const { return fitpars_.rinv(); }
  double phi0fit() const { return fitpars_.phi0(); }
  double d0fit() const { return fitpars_.d0(); }
  double tfit() const { return fitpars_.t(); }
  double z0fit() const { return fitpars_.z0(); }
  double chiSqfit() const { return chisqfit_; }

  double rinvfitexact() const { return fitparsexact_.rinv(); }
  double phi0fitexact() const { return fitparsexact_.phi0(); }
  double d0fitexact() const { return fitparsexact_.d0(); }
  double tfitexact() const { return fitparsexact_.t(); }
  double z0fitexact() const { return fitparsexact_.z0(); }

  FPGAWord irinvfit() const { return fpgafitpars_.rinv(); }
  FPGAWord iphi0fit() const { return fpgafitpars_.phi0(); }
  FPGAWord id0fit() const { return fpgafitpars_.d0(); }
  FPGAWord itfit() const { return fpgafitpars_.t(); }
  FPGAWord iz0fit() const { return fpgafitpars_.z0(); }
  FPGAWord ichiSqfit() const { return ichisqfit_; }

  void setFitPars(double rinvfit, double phi0fit, double d0fit, double tfit,
		  double z0fit, double chisqfit,
		  double rinvfitexact, double phi0fitexact, double d0fitexact, double tfitexact,
		  double z0fitexact, double chisqfitexact,
		  int irinvfit, int iphi0fit, int id0fit, int itfit,
		  int iz0fit, int ichisqfit,
		  const vector<L1TStub*>& l1stubs = vector<L1TStub*>()){

    fitpars_.init(rinvfit,phi0fit,d0fit,tfit,z0fit);
    chisqfit_=chisqfit;

    fitparsexact_.init(rinvfitexact,phi0fitexact,d0fitexact,tfitexact,z0fitexact);
    chisqfitexact_=chisqfitexact;
    
    if (irinvfit>(1<<14)) irinvfit=(1<<14);
    if (irinvfit<=-(1<<14)) irinvfit=-(1<<14)+1;
    fpgafitpars_.rinv().set(irinvfit,15,false,__LINE__,__FILE__);
    fpgafitpars_.phi0().set(iphi0fit,19,false,__LINE__,__FILE__);
    fpgafitpars_.d0().set(id0fit,19,false,__LINE__,__FILE__);
    fpgafitpars_.t().set(itfit,14,false,__LINE__,__FILE__);

    if (iz0fit>=(1<<(nbitsz0-1))) {
      iz0fit=(1<<(nbitsz0-1))-1; 
    }

    if (iz0fit<=-(1<<(nbitsz0-1))) {
      iz0fit=1-(1<<(nbitsz0-1)); 
    }
    
    fpgafitpars_.z0().set(iz0fit,nbitsz0,false,__LINE__,__FILE__);
    ichisqfit_.set(ichisqfit,8,true,__LINE__,__FILE__);

    delete fpgatrack_;
    fpgatrack_=new Track(makeTrack(l1stubs));

  }
  
  std::string trackfitstr() {
    std::ostringstream oss;

    string stubid0="111111111";
    string stubid1="111111111";
    string stubid2="111111111";
    string stubid3="111111111";

    if (isBarrel()) {
      if (layer()==1) {
	if (layerresid_[2].valid()) {
	  stubid0=layerresid_[2].fpgastubid().str();
	}
	if (layerresid_[3].valid()) {
	  stubid1=layerresid_[3].fpgastubid().str();
	}
	if (layerresid_[4].valid()) {
	  stubid2=layerresid_[4].fpgastubid().str();
	}
	if (layerresid_[5].valid()) {
	  stubid3=layerresid_[5].fpgastubid().str();
	}
	if (diskresid_[0].valid()) {
	  stubid3=diskresid_[0].fpgastubid().str();
	}
	if (diskresid_[1].valid()) {
	  stubid2=diskresid_[1].fpgastubid().str();
	}
	if (diskresid_[2].valid()) {
	  stubid1=diskresid_[2].fpgastubid().str();
	}
	if (diskresid_[3].valid()) {
	  stubid0=diskresid_[3].fpgastubid().str();
	}
      }
      
      if (layer()==3) {
	if (layerresid_[0].valid()) {
	  stubid0=layerresid_[0].fpgastubid().str();
	}
	if (layerresid_[1].valid()) {
	  stubid1=layerresid_[1].fpgastubid().str();
	}
	if (layerresid_[4].valid()) {
	  stubid2=layerresid_[4].fpgastubid().str();
	}
	if (layerresid_[5].valid()) {
	  stubid3=layerresid_[5].fpgastubid().str();
	}
	if (diskresid_[0].valid()) {
	  stubid3=diskresid_[0].fpgastubid().str();
	}
	if (diskresid_[1].valid()) {
	  stubid2=diskresid_[1].fpgastubid().str();
	}
      }
      
      if (layer()==5) {
	if (layerresid_[0].valid()) {
	  stubid0=layerresid_[0].fpgastubid().str();
	}
	if (layerresid_[1].valid()) {
	  stubid1=layerresid_[1].fpgastubid().str();
	}
	if (layerresid_[2].valid()) {
	  stubid2=layerresid_[2].fpgastubid().str();
	}
	if (layerresid_[3].valid()) {
	  stubid3=layerresid_[3].fpgastubid().str();
	}
      }
    }
    
    if (isDisk()) {
      if (disk()==1) {
	if (layerresid_[0].valid()) {
	  stubid0=layerresid_[0].fpgastubid().str();
	}
	if (diskresid_[2].valid()) {
	  stubid1=diskresid_[2].fpgastubid().str();
	}
	if (diskresid_[3].valid()) {
	  stubid2=diskresid_[3].fpgastubid().str();
	}
	if (diskresid_[4].valid()) {
	  stubid3=diskresid_[4].fpgastubid().str();
	} else  if (layerresid_[1].valid()) {
	  stubid3=layerresid_[1].fpgastubid().str();
	}
      }

      if (disk()==3) {
	if (layerresid_[0].valid()) {
	  stubid0=layerresid_[0].fpgastubid().str();
	}
	if (diskresid_[0].valid()) {
	  stubid1=diskresid_[0].fpgastubid().str();
	}
	if (diskresid_[1].valid()) {
	  stubid2=diskresid_[1].fpgastubid().str();
	}
	if (diskresid_[4].valid()) {
	  stubid3=diskresid_[4].fpgastubid().str();
	} else  if (layerresid_[1].valid()) {
	  stubid3=layerresid_[1].fpgastubid().str();
	}
      }
      
    }
    
    if (isOverlap()) {
      if (layer()==1) {
	if (diskresid_[1].valid()) {
	  stubid0=diskresid_[1].fpgastubid().str();
	}
	if (diskresid_[2].valid()) {
	  stubid1=diskresid_[2].fpgastubid().str();
	}
	if (diskresid_[3].valid()) {
	  stubid2=diskresid_[3].fpgastubid().str();
	}
	if (diskresid_[4].valid()) {
	  stubid3=diskresid_[4].fpgastubid().str();
	}
	
      }
      
    }
      


    
    
    
    // real Q print out for fitted tracks
    if(writeoutReal){
      oss << (fpgafitpars_.rinv().value())*krinvpars<<" "
	  << (fpgafitpars_.phi0().value())*kphi0pars<<" "
	  << (fpgafitpars_.d0().value())*kd0pars<<" "
	  << (fpgafitpars_.t().value())*ktpars<<" "
	  << (fpgafitpars_.z0().value())*kz<<" "
      //<< ichisqfit_.str()<< "|"                            
        << innerFPGAStub_->phiregionaddressstr()<<" ";
    if (middleFPGAStub_) {
      oss << middleFPGAStub_->phiregionaddressstr()<<" ";
    }
    oss << outerFPGAStub_->phiregionaddressstr()<<" "
	<< stubid0<<"|"
	<< stubid1<<"|"
	<< stubid2<<"|"
	<< stubid3;
    }
    //Binary print out
    if(!writeoutReal){
      oss << fpgafitpars_.rinv().str()<<"|"
	  << fpgafitpars_.phi0().str()<<"|"
	  << fpgafitpars_.d0().str()<<"|"
	//<< "xxxxxxxxxxx|"
	  << fpgafitpars_.t().str()<<"|"
	  << fpgafitpars_.z0().str()<<"|"
	//<< ichisqfit_.str()<< "|"
	  << innerFPGAStub_->phiregionaddressstr()<<"|";
    if (middleFPGAStub_) {
      oss << middleFPGAStub_->phiregionaddressstr()<<"|";
    }
      oss << outerFPGAStub_->phiregionaddressstr()<<"|"
	  << stubid0<<"|"
	  << stubid1<<"|"
	  << stubid2<<"|"
	  << stubid3;
    }
    return oss.str();
  }
  
  
  Track makeTrack(vector<L1TStub*> l1stubs) {
    assert(fit());
    if (l1stubs.size() == 0) {
      l1stubs = getL1Stubs(); // If fitter produced no stub list, take it from original tracklet.
    };
    Track tmpTrack(fpgafitpars_.rinv().value(),
		   fpgafitpars_.phi0().value(),
		   fpgafitpars_.d0().value(),
		   fpgafitpars_.t().value(),
		   fpgafitpars_.z0().value(),
		   ichisqfit_.value(),
		   chisqfit_,
		   getStubIDs(),
		   l1stubs,
		   getISeed());

    return tmpTrack;
    
  }

  Track* getTrack() {
    assert(fpgatrack_!=0);
    return fpgatrack_;
  }
  

  bool fit() const { return ichisqfit_.value()!=-1; }

  int layer() const {
    int l1 = (innerFPGAStub_ && innerFPGAStub_->isBarrel()) ? innerStub_->layer()+1 : 999,
        l2 = (middleFPGAStub_ && middleFPGAStub_->isBarrel()) ? middleStub_->layer()+1 : 999,
        l3 = (outerFPGAStub_ && outerFPGAStub_->isBarrel()) ? outerStub_->layer()+1 : 999,
        l = min(min(l1,l2),l3);
    return (l < 999 ? l : 0);
  }
  
  int disk() const {
    int d1 = (innerFPGAStub_ && innerFPGAStub_->isDisk()) ? innerStub_->disk() : 999,
        d2 = (middleFPGAStub_ && middleFPGAStub_->isDisk()) ? middleStub_->disk() : 999,
        d3 = (outerFPGAStub_ && outerFPGAStub_->isDisk()) ? outerStub_->disk() : 999,
        d = 999;
    if (abs(d1) < min(abs(d2),abs(d3))) d = d1;
    if (abs(d2) < min(abs(d1),abs(d3))) d = d2;
    if (abs(d3) < min(abs(d1),abs(d2))) d = d3;
    return (d < 999 ? d : 0);
  }

  int disk2() const {
    if (innerStub_->disk()>0) {
      return innerStub_->disk()+1;
    }
    return innerStub_->disk()-1;
  }

  int overlap() const {
    return innerStub_->layer()+21;
  }

  bool isBarrel() const { 
    return barrel_;
  }

  bool isOverlap() const { 
    return overlap_;
  }

  int isDisk() const { 
    return disk_;
  }

  bool foundTrack(L1SimTrack simtrk,double phioffset){

    double deta=simtrk.eta()-asinh(itfit().value()*ktpars);
    double dphi=Util::phiRange(simtrk.phi()-(iphi0fit().value()*kphi0pars+phioffset));
    
    bool found=(fabs(deta)<0.06)&&(fabs(dphi)<0.01);
    
    return found;

  }

  void setTrackletIndex(int index) {
    trackletIndex_=index;
    assert(index<128);
  }

  int trackletIndex() const {return trackletIndex_;}

  void setTCIndex(int index) {TCIndex_=index;}

  int TCIndex() const {return TCIndex_;}
  
  int TCID() const {
    return TCIndex_*(1<<7)+trackletIndex_;
  }

  int getISeed() const {
    int iSeed = TCIndex_ >> 4;
    assert(iSeed >= 0 && iSeed <= 11);
    return iSeed;
  }

  int getITC() const {
    int iSeed = getISeed(),
        iTC = TCIndex_ - (iSeed << 4);
    assert(iTC >= 0 && iTC <= 14);
    return iTC;
  }
  
  unsigned int PSseed() {
    return ((layer()==1)||(disk()!=0))?1:0;
  }

  unsigned int seedIndex() const {
    return seedIndex_;
  }


  //should be removed and use seedindex instead
  int seed() const {
    // Returns integer code for tracklet seed.
    // Barrel: L1L2=1, L2L3=2, L3L4=3, L5L6=6
    // Disk: D1D2=11, D3D4=13 (+/- for forward/backward disk)
    // Overlap: L1D1=21, L2D1=22 (+/- for forward/backward disk)
    if(barrel_) return innerStub_->layer()+1;
    if(disk_) {
      if(innerStub_->disk() < 0) return innerStub_->disk()-10;
      else return innerStub_->disk()+10;
    }
    if(overlap_) {
      if(innerStub_->disk() < 0) return -21-outerStub_->layer();
      else return 21+outerStub_->layer();
    }
    return 0;
  }


  
  unsigned int calcSeedIndex() const {
  
    int seedindex=-1;
    int seedlayer=layer();
    int seeddisk=disk();
    
    if (seedlayer==1&&seeddisk==0) seedindex=0;  //L1L2
    if (seedlayer==3&&seeddisk==0) seedindex=1;  //L3L4
    if (seedlayer==5&&seeddisk==0) seedindex=2;  //L5L6
    if (seedlayer==0&&abs(seeddisk)==1) seedindex=3;  //D1D2
    if (seedlayer==0&&abs(seeddisk)==3) seedindex=4;  //D3D4
    if (seedlayer==1&&abs(seeddisk)==1) seedindex=5;  //L1D1
    if (seedlayer==2&&abs(seeddisk)==1) seedindex=6;  //L2D1
    if (seedlayer==2&&abs(seeddisk)==0) seedindex=7;  //L2L3
    if (middleFPGAStub_&&seedlayer==2&&seeddisk==0) seedindex = 8; // L3L4L2
    if (middleFPGAStub_&&seedlayer==4&&seeddisk==0) seedindex = 9; // L5L6L4
    assert(innerFPGAStub_!=0);
    assert(outerFPGAStub_!=0);
    if (middleFPGAStub_&&seedlayer==2&&abs(seeddisk)==1) {
      int l1 = (innerFPGAStub_ && innerFPGAStub_->isBarrel()) ? innerStub_->layer()+1 : 999,
          l2 = (middleFPGAStub_ && middleFPGAStub_->isBarrel()) ? middleStub_->layer()+1 : 999,
          l3 = (outerFPGAStub_ && outerFPGAStub_->isBarrel()) ? outerStub_->layer()+1 : 999;
      if (l1+l2+l3 < 1998) { // If two stubs are layer stubs
        seedindex = 10; // L2L3D1
      } else {
        seedindex = 11; // D1D2L2
      }
    }

    if (seedindex<0) {
      cout << "seedlayer abs(seeddisk) : "<<seedlayer<<" "<<abs(seeddisk)<<endl;
      assert(0);
    }

    return seedindex;
  }

private:

  //Three types of tracklets... Overly complicated
  bool barrel_;
  bool disk_;
  bool overlap_;
  bool triplet_;

  Stub* innerFPGAStub_;
  Stub* middleFPGAStub_;
  Stub* outerFPGAStub_;


  L1TStub* innerStub_;
  L1TStub* middleStub_;
  L1TStub* outerStub_;

  int trackletIndex_;
  int TCIndex_;

  unsigned int seedIndex_;

  //Tracklet track parameters  

  TrackPars<FPGAWord> fpgapars_;

  TrackPars<double> trackpars_;
  
  TrackPars<double> trackparsapprox_;

  int      projlayer_[4];
  int      projdisk_[5];

  //Track  parameters from track fit

  TrackPars<FPGAWord> fpgafitpars_;  
  FPGAWord ichisqfit_;

  TrackPars<double> fitpars_;  
  double chisqfit_;

  TrackPars<double> fitparsexact_;  
  double chisqfitexact_;

  Track *fpgatrack_;


  LayerProjection layerproj_[6];
  DiskProjection diskproj_[5];

  LayerResidual layerresid_[6];
  DiskResidual diskresid_[5];

  
};



#endif

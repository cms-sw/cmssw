#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"

using namespace std;
using namespace trklet;


TrackletEngineUnit::TrackletEngineUnit(const Settings& settings, unsigned int nbitsfinephi, unsigned int layerdisk2, unsigned int iSeed, unsigned int nbitsfinephidiff, unsigned int iAllStub, std::vector<bool> pttableinner, std::vector<bool> pttableouter, VMStubsTEMemory* outervmstubs)
  : settings_(settings), candpairs_(5) {
  idle_=true;
  nbitsfinephi_=nbitsfinephi;
  layerdisk2_=layerdisk2;
  iSeed_=iSeed;
  nbitsfinephidiff_=nbitsfinephidiff;
  iAllStub_=iAllStub;
  pttableinner_=pttableinner;
  pttableouter_=pttableouter;
  outervmstubs_=outervmstubs;
}


void TrackletEngineUnit::init(const TEData& tedata){
  tedata_=tedata;
  nreg_=0;
  istub_=0;
  idle_=false;
  assert(tedata_.regions_.size()>0);
  std::tie(next_,ireg_,nstub_) = tedata_.regions_[0];
}

void TrackletEngineUnit::reset(){
  idle_=true;
  candpairs_.reset();
}

void TrackletEngineUnit::step(){
  if (candpairs_.full())
    return;


  int ibin=tedata_.start_+next_;

  int nbins=8; //FIXME
  
  assert(istub_<outervmstubs_->nVMStubsBinned(ireg_*nbins+ibin));
	  
  const VMStubTE& outervmstub = outervmstubs_->getVMStubTEBinned(ireg_*nbins+ibin,istub_);
  int rzbin = (outervmstub.vmbits().value() & 7);
	  
  //assert(innerphibits_ != -1);
  //assert(outerphibits_ != -1);
	  
  FPGAWord iphiouterbin = outervmstub.finephi();

  assert(iphiouterbin == outervmstub.finephi());
	
  //New code to calculate lut value
  int outerfinephi=iAllStub_*(1<<(nbitsfinephi_-settings_.nbitsallstubs(layerdisk2_)))+ireg_*(1<<settings_.nfinephi(1,iSeed_))+iphiouterbin.value();
  int idphi=outerfinephi-tedata_.innerfinephi_;
  bool inrange=(idphi<(1<<(nbitsfinephidiff_-1)))&&(idphi>=-(1<<(nbitsfinephidiff_-1)));
  //int idphiraw=idphi;
  if (idphi<0) idphi=idphi+(1<<nbitsfinephidiff_);

  if (iSeed_ >= 4) {  //Also use r-position
    int ir = ((ibin & 3) << 1) + (rzbin >> 2);
    idphi = (idphi << 3) + ir;
  }
	
  if (next_!=0)
    rzbin += 8;
  if ((rzbin < tedata_.rzbinfirst_) || (rzbin - tedata_.rzbinfirst_ > tedata_.rzdiffmax_)) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet")
	<< " layer-disk stub pair rejected because rbin cut : " << rzbin << " " << tedata_.rzbinfirst_
	<< " " << tedata_.rzdiffmax_;
    }
  } else {

    FPGAWord outerbend = outervmstub.bend();
	
    int ptinnerindex = (idphi << tedata_.innerbend_.nbits()) + tedata_.innerbend_.value();
    int ptouterindex = (idphi << outerbend.nbits()) + outerbend.value();
    
    if (!(inrange && pttableinner_[ptinnerindex] && pttableouter_[ptouterindex])) {
      if (settings_.debugTracklet()) {
	edm::LogVerbatim("Tracklet") <<" Stub pair rejected because of stub pt cut bends : "
				     << benddecode(tedata_.innerbend_.value(), tedata_.stub_->isPSmodule()) << " "
				     << benddecode(outerbend.value(), outervmstub.isPSmodule());
      }
    } else {
      candpairs_.store(pair<const Stub*, const Stub*>(tedata_.stub_, outervmstub.stub()));
    }
  }
  istub_++;
  if (istub_>=nstub_) {
    istub_=0;
    nreg_++;
    if (nreg_>=tedata_.regions_.size()) {
      nreg_=0;
      idle_=true;
    } else {
      std::tie(next_,ireg_,nstub_) = tedata_.regions_[nreg_];
    }
  }
  
}


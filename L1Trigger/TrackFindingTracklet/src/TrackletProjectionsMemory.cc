#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

TrackletProjectionsMemory::TrackletProjectionsMemory(string name, Settings const& settings)
    : MemoryBase(name, settings) {
  size_t pos = find_nth(name, 0, "_", 1);
  assert(pos != string::npos);
  initLayerDisk(pos + 1, layer_, disk_);
  hasProj_ = false;
  npage_ = name.size()-17;
  if (name.substr(name.size()-2,2)=="_E") {
    npage_ = name.size()-19;
  }
  tracklets_.resize(npage_);
}

void TrackletProjectionsMemory::addProj(Tracklet* tracklet, unsigned int page) {
  if (layer_ != 0 && disk_ == 0)
    assert(tracklet->validProj(layer_ - 1));
  if (layer_ == 0 && disk_ != 0)
    assert(tracklet->validProj(N_LAYER + abs(disk_) - 1));
  if (layer_ != 0 && disk_ != 0)
    assert(tracklet->validProj(layer_ - 1) || tracklet->validProj(N_LAYER + abs(disk_) - 1));

  for (auto& itracklet : tracklets_[page]) {
    if (itracklet == tracklet) {
      edm::LogPrint("Tracklet") << "Adding same tracklet " << tracklet << " twice in " << getName();
    }
    assert(itracklet != tracklet);
  }

  hasProj_ = true;

  if (tracklets_[page].size()<(1<<(N_BITSMEMADDRESS-1))-1) {
    tracklets_[page].push_back(tracklet);
  }
}

void TrackletProjectionsMemory::clean() {
  for (unsigned int i = 0; i < tracklets_.size() ; i++){
    tracklets_[i].clear(); 
  }
}

void TrackletProjectionsMemory::writeTPROJ(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirTP = settings_.memPath() + "TrackletProjections/";


  //Hack to suppress writing empty TPROJ memories - only want to write MPROJ memories
  if (getName()[0] == 'T') return;
  
  std::ostringstream oss;
  oss << dirTP << "TrackletProjections_" << getName() << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1)
      << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirTP, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int j = 0; j < tracklets_.size(); j++) {
    // This is a hack here to write out the TPAR files for backward compatibility
    std::ofstream out;
    std::string moduleName = getName().substr(0,10);;
    moduleName[0]='T';
    std::ostringstream oss2;
    char postfix = getName()[10];
    postfix+=j;
    oss2 << dirTP << "TrackletProjections_" << moduleName << postfix << "_" << getName().substr(getName().size()-6,6)
	 << "_" <<std::setfill('0') << std::setw(2) << (iSector_ + 1)
	 << ".dat";
    std::string fnameTPAR = oss2.str();
    openfile(out, first, dirTP, fnameTPAR, __FILE__, __LINE__);
    out << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;
    for (unsigned int i = 0; i < tracklets_[j].size(); i++) {
    
      string proj = (layer_ > 0 && tracklets_[j][i]->validProj(layer_ - 1)) ? tracklets_[j][i]->trackletprojstrlayer(layer_)
	: tracklets_[j][i]->trackletprojstrdisk(disk_);
      out_ << hexstr(j) << " " << hexstr(i) << " " << proj << "  " << trklet::hexFormat(proj) << endl;
      out << hexstr(i) << " " << proj << "  " << trklet::hexFormat(proj) << endl;
    }
    out.close();
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}

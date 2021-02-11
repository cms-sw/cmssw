#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

VMStubsMEMemory::VMStubsMEMemory(string name, Settings const& settings, unsigned int iSector)
    : MemoryBase(name, settings, iSector) {
  unsigned int layerdisk = initLayerDisk(6);
  if (layerdisk < N_LAYER) {
    binnedstubs_.resize(settings_.NLONGVMBINS());
  } else {
    //For disks we have NLONGVMBITS on each disk
    binnedstubs_.resize(2 * settings_.NLONGVMBINS());
  }
}

void VMStubsMEMemory::writeStubs(bool first) {
  const string dirVM = settings_.memPath() + "VMStubsME/";

  std::ostringstream oss;
  oss << dirVM << "VMStubs_" << getName();
  //get rid of duplicates
  auto const& tmp = oss.str();
  int len = tmp.size();
  if (tmp[len - 2] == 'n' && tmp[len - 1] > '1' && tmp[len - 1] <= '9')
    return;
  oss << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  if (first) {
    bx_ = 0;
    event_ = 1;

    if (not std::filesystem::exists(dirVM)) {
      int fail = system((string("mkdir -p ") + dirVM).c_str());
      if (fail)
        throw cms::Exception("BadDir") << __FILE__ << " " << __LINE__ << " could not create directory " << dirVM;
    }
    out_.open(fname);
    if (out_.fail())
      throw cms::Exception("BadFile") << __FILE__ << " " << __LINE__ << " could not create file " << fname;

  } else
    out_.open(fname, std::ofstream::app);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int i = 0; i < binnedstubs_.size(); i++) {
    for (unsigned int j = 0; j < binnedstubs_[i].size(); j++) {
      string stub = binnedstubs_[i][j].stubindex().str();
      stub += "|" + binnedstubs_[i][j].bend().str();

      FPGAWord finepos = binnedstubs_[i][j].finerz();
      stub += "|" + finepos.str();
      out_ << hex << i << " " << j << dec << " " << stub << " " << trklet::hexFormat(stub) << endl;
    }
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}

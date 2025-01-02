#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

VMStubsMEMemory::VMStubsMEMemory(string name, Settings const& settings) : MemoryBase(name, settings) {
  layerdisk_ = initLayerDisk(6);
  if (layerdisk_ < N_LAYER) {
    binnedstubs_.resize(settings_.NLONGVMBINS());
  } else {
    //For disks we have NLONGVMBITS on each disk
    binnedstubs_.resize(2 * settings_.NLONGVMBINS());
  }
}

void VMStubsMEMemory::writeStubs(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirVM = settings_.memPath() + "VMStubsME/";

  std::ostringstream oss;
  oss << dirVM << "VMStubs_" << getName();
  oss << "_" << std::setfill('0') << std::setw(2) << (iSector_ + 1) << ".dat";
  auto const& fname = oss.str();

  openfile(out_, first, dirVM, fname, __FILE__, __LINE__);

  out_ << "BX = " << (bitset<3>)bx_ << " Event : " << event_ << endl;

  for (unsigned int i = 0; i < binnedstubs_.size(); i++) {
    int nbitsrz = (layerdisk_ < N_LAYER) ? 3 : 4;
    unsigned int newi = 8*(i&((1<<nbitsrz)-1)) + (i>>nbitsrz);
    for (unsigned int j = 0; j < binnedstubs_[i].size(); j++) {
      string stub = binnedstubs_[i][j].stubindex().str();
      stub += "|" + binnedstubs_[i][j].bend().str();

      FPGAWord finephipos = binnedstubs_[i][j].finephi();
      stub += "|" + finephipos.str();
      FPGAWord finepos = binnedstubs_[i][j].finerz();
      stub += "|" + finepos.str();

      out_ << hexstr(newi) << " " << hexstr(j) << " " << stub << " " << trklet::hexFormat(stub) << endl;
    }
  }
  out_.close();

  bx_++;
  event_++;
  if (bx_ > 7)
    bx_ = 0;
}

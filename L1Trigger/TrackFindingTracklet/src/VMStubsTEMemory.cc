#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>
#include <filesystem>

using namespace std;
using namespace trklet;

VMStubsTEMemory::VMStubsTEMemory(string name, Settings const& settings)
    : MemoryBase(name, settings), bendtable_(settings) {
  //set the layer or disk that the memory is in
  initLayerDisk(6, layer_, disk_);

  layerdisk_ = initLayerDisk(6);

  //Pointer to other VMStub memory for creating stub pairs
  other_ = nullptr;

  //What type of seeding is this memory used for
  initSpecialSeeding(11, overlap_, extra_, extended_);

  string subname = name.substr(12, 2);
  phibin_ = subname[0] - '0';
  if (subname[1] != 'n') {
    phibin_ = 10 * phibin_ + (subname[1] - '0');
  }

  isinner_ = (layer_ % 2 == 1 or disk_ % 2 == 1);
  // special cases with overlap seeding
  if (overlap_ and layer_ == 2)
    isinner_ = true;
  if (overlap_ and layer_ == 3)
    isinner_ = false;
  if (overlap_ and disk_ == 1)
    isinner_ = false;

  if (extra_ and layer_ == 2)
    isinner_ = true;
  if (extra_ and layer_ == 3)
    isinner_ = false;
  // more special cases for triplets
  if (!overlap_ and extended_ and layer_ == 2)
    isinner_ = true;
  if (!overlap_ and extended_ and layer_ == 3)
    isinner_ = false;
  if (overlap_ and extended_ and layer_ == 2)
    isinner_ = false;
  if (overlap_ and extended_ and disk_ == 1)
    isinner_ = false;

  stubsbinnedvm_.resize(settings_.NLONGVMBINS());
}

bool VMStubsTEMemory::addVMStub(VMStubTE vmstub, int bin) {
  //If the pt of the stub is consistent with the allowed pt of tracklets
  //in that can be formed in this VM and the other VM used in the TE.

  if (settings_.combined()) {
    if (disk_ > 0) {
      assert(vmstub.stub()->isPSmodule());
    }
    bool negdisk = vmstub.stub()->disk().value() < 0.0;
    if (negdisk)
      bin += 4;
    assert(bin < (int)stubsbinnedvm_.size());
    if (stubsbinnedvm_[bin].size() < N_VMSTUBSMAX) {
      stubsbinnedvm_[bin].push_back(vmstub);
      stubsvm_.push_back(vmstub);
    }
    return true;
  }

  bool pass = false;
  if (settings_.extended() && bendtable_.size() == 0) {
    pass = true;
  } else {
    pass = bendtable_.lookup(vmstub.bend().value());
  }

  if (!pass) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << getName() << " Stub failed bend cut. bend = "
                                   << settings_.benddecode(vmstub.bend().value(), layerdisk_, vmstub.isPSmodule());
    return false;
  }

  bool negdisk = vmstub.stub()->disk().value() < 0.0;

  if (overlap_) {
    if (disk_ == 1) {
      assert(bin < 4);
      if (negdisk)
        bin += 4;
      if (stubsbinnedvm_[bin].size() >= settings_.maxStubsPerBin())
        return false;
      stubsbinnedvm_[bin].push_back(vmstub);
      if (settings_.debugTracklet())
        edm::LogVerbatim("Tracklet") << getName() << " Stub in disk = " << disk_ << "  in bin = " << bin;
    } else if (layer_ == 2) {
      if (stubsbinnedvm_[bin].size() >= settings_.maxStubsPerBin())
        return false;
      stubsbinnedvm_[bin].push_back(vmstub);
    }
  } else {
    if (vmstub.stub()->layerdisk() < N_LAYER) {
      if (!isinner_) {
        if (stubsbinnedvm_[bin].size() >= settings_.maxStubsPerBin())
          return false;
        stubsbinnedvm_[bin].push_back(vmstub);
      }

    } else {
      if (disk_ % 2 == 0) {
        assert(bin < 4);
        if (negdisk)
          bin += 4;
        if (stubsbinnedvm_[bin].size() >= settings_.maxStubsPerBin())
          return false;
        stubsbinnedvm_[bin].push_back(vmstub);
      }
    }
  }
  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "Adding stubs to " << getName();
  if (stubsbinnedvm_[bin].size() >= settings_.maxStubsPerBin())
    return false;
  stubsvm_.push_back(vmstub);
  return true;
}

// TODO - should migrate away from using this method for any binned memory
bool VMStubsTEMemory::addVMStub(VMStubTE vmstub) {
  FPGAWord binlookup = vmstub.vmbits();

  assert(binlookup.value() >= 0);
  int bin = (binlookup.value() / 8);

  //If the pt of the stub is consistent with the allowed pt of tracklets
  //in that can be formed in this VM and the other VM used in the TE.

  bool pass = false;
  if (settings_.extended() && bendtable_.size() == 0) {
    pass = true;
  } else {
    pass = bendtable_.lookup(vmstub.bend().value());
  }

  if (!pass) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << getName() << " Stub failed bend cut. bend = "
                                   << settings_.benddecode(vmstub.bend().value(), layerdisk_, vmstub.isPSmodule());
    return false;
  }

  bool negdisk = vmstub.stub()->disk().value() < 0.0;

  if (!extended_) {
    if (overlap_) {
      if (disk_ == 1) {
        assert(bin < 4);
        if (negdisk)
          bin += 4;
        stubsbinnedvm_[bin].push_back(vmstub);
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " Stub with lookup = " << binlookup.value()
                                       << " in disk = " << disk_ << "  in bin = " << bin;
        }
      }
    } else {
      if (vmstub.stub()->layerdisk() < N_LAYER) {
        if (!isinner_) {
          stubsbinnedvm_[bin].push_back(vmstub);
        }

      } else {
        if (disk_ % 2 == 0) {
          assert(bin < 4);
          if (negdisk)
            bin += 4;
          stubsbinnedvm_[bin].push_back(vmstub);
        }
      }
    }
  } else {  //extended
    if (!isinner_) {
      if (layer_ > 0) {
        stubsbinnedvm_[bin].push_back(vmstub);
      } else {
        if (overlap_) {
          assert(disk_ == 1);  // D1 from L2L3D1

          //bin 0 is PS, 1 through 3 is 2S
          if (vmstub.stub()->isPSmodule()) {
            bin = 0;
          } else {
            bin = vmstub.stub()->r().value();  // 0 to 9
            bin = bin >> 2;                    // 0 to 2
            bin += 1;
          }
        }
        assert(bin < 4);
        if (negdisk)
          bin += 4;
        stubsbinnedvm_[bin].push_back(vmstub);
      }
    }
  }

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "Adding stubs to " << getName();
  stubsvm_.push_back(vmstub);
  return true;
}

void VMStubsTEMemory::clean() {
  stubsvm_.clear();
  for (auto& stubsbinnedvm : stubsbinnedvm_) {
    stubsbinnedvm.clear();
  }
}

void VMStubsTEMemory::writeStubs(bool first, unsigned int iSector) {
  iSector_ = iSector;
  const string dirVM = settings_.memPath() + "VMStubsTE/";
  openFile(first, dirVM, "VMStubs_");

  if (isinner_) {  // inner VM for TE purpose
    for (unsigned int j = 0; j < stubsvm_.size(); j++) {
      out_ << "0x";
      out_ << std::setfill('0') << std::setw(2);
      out_ << hex << j << dec;
      string stub = stubsvm_[j].str();
      out_ << " " << stub << " " << trklet::hexFormat(stub) << endl;
    }
  } else {  // outer VM for TE purpose
    for (unsigned int i = 0; i < stubsbinnedvm_.size(); i++) {
      for (unsigned int j = 0; j < stubsbinnedvm_[i].size(); j++) {
        string stub = stubsbinnedvm_[i][j].str();
        out_ << hex << i << " " << j << dec << " " << stub << " " << trklet::hexFormat(stub) << endl;
      }
    }
  }

  out_.close();
}

void VMStubsTEMemory::getPhiRange(double& phimin, double& phimax, unsigned int iSeed, unsigned int inner) {
  int nvm = -1;
  if (overlap_) {
    if (layer_ > 0) {
      nvm = settings_.nallstubs(layer_ - 1) * settings_.nvmte(inner, iSeed);
    }
    if (disk_ > 0) {
      nvm = settings_.nallstubs(disk_ + N_DISK) * settings_.nvmte(inner, iSeed);
    }
  } else {
    if (layer_ > 0) {
      nvm = settings_.nallstubs(layer_ - 1) * settings_.nvmte(inner, iSeed);
      if (extra_) {
        nvm = settings_.nallstubs(layer_ - 1) * settings_.nvmte(inner, iSeed);
      }
    }
    if (disk_ > 0) {
      nvm = settings_.nallstubs(disk_ + N_DISK) * settings_.nvmte(inner, iSeed);
    }
  }
  assert(nvm > 0);
  assert(nvm <= 32);
  double dphi = settings_.dphisectorHG() / nvm;
  phimax = phibin() * dphi;
  phimin = phimax - dphi;

  return;
}

void VMStubsTEMemory::setbendtable(const TrackletLUT& bendtable) { bendtable_ = bendtable; }

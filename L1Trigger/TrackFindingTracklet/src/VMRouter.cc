#include "L1Trigger/TrackFindingTracklet/interface/VMRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"
#include "L1Trigger/TrackFindingTracklet/interface/InputLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

VMRouter::VMRouter(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global),
      meTable_(settings),
      diskTable_(settings),
      innerTable_(settings),
      innerOverlapTable_(settings),
      innerThirdTable_(settings) {
  layerdisk_ = initLayerDisk(4);

  vmstubsMEPHI_.resize(settings_.nvmme(layerdisk_), nullptr);

  unsigned int region = name[9] - 'A';
  assert(region < settings_.nallstubs(layerdisk_));

  overlapbits_ = 7;
  nextrabits_ = overlapbits_ - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_));

  meTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::me, region);  //used for ME and outer TE barrel

  if (layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D2 || layerdisk_ == D4) {
    diskTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::disk, region);  //outer disk used by D1, D2, and D4
  }

  if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3 ||
      layerdisk_ == LayerDisk::L5 || layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3) {
    innerTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::inner, region);  //projection to next layer/disk
  }

  if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2) {
    innerOverlapTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::inneroverlap, region);  //projection to disk from layer
  }

  if (layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L5 ||
      layerdisk_ == LayerDisk::D1) {
    innerThirdTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::innerthird, region);  //projection to third layer/disk
  }

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk_);
}

void VMRouter::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }

  if (output.substr(0, 10) == "allstubout") {
    AllStubsMemory* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    allstubs_.push_back(tmp);
    return;
  }

  if (output.substr(0, 12) == "vmstuboutPHI" || output.substr(0, 14) == "vmstuboutMEPHI" ||
      output.substr(0, 15) == "vmstuboutTEIPHI" || output.substr(0, 15) == "vmstuboutTEOPHI") {
    char seedtype = memory->getName().substr(11, 1)[0];
    unsigned int pos = 12;
    int vmbin = memory->getName().substr(pos, 1)[0] - '0';
    pos++;
    if (pos < memory->getName().size()) {
      if (memory->getName().substr(pos, 1)[0] != 'n') {
        vmbin = vmbin * 10 + memory->getName().substr(pos, 1)[0] - '0';
        pos++;
      }
    }

    int iseed = -1;
    unsigned int inner = 1;
    if (memory->getName().substr(3, 2) == "TE") {
      VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
      assert(tmp != nullptr);
      if (seedtype < 'I') {
        if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2)
          iseed = Seed::L1L2;
        if (layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L4)
          iseed = Seed::L3L4;
        if (layerdisk_ == LayerDisk::L5 || layerdisk_ == LayerDisk::L6)
          iseed = Seed::L5L6;
        if (layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D2)
          iseed = Seed::D1D2;
        if (layerdisk_ == LayerDisk::D3 || layerdisk_ == LayerDisk::D4)
          iseed = Seed::D3D4;
        if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L5 ||
            layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3)
          inner = 0;
      } else if (seedtype < 'M') {
        if (layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3)
          iseed = Seed::L2L3;
        if (layerdisk_ == LayerDisk::L2)
          inner = 0;
      } else if (seedtype <= 'Z') {
        if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::D1)
          iseed = Seed::L1D1;
        if (layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::D1)
          iseed = Seed::L2D1;
        if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2)
          inner = 0;
      } else if (seedtype < 'o' && seedtype >= 'a') {
        if (layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3)
          iseed = Seed::L2L3D1;
        if (layerdisk_ == LayerDisk::L2)
          inner = 0;
      } else if (seedtype > 'o' && seedtype <= 'z') {
        if (layerdisk_ == LayerDisk::L2)
          iseed = Seed::D1D2L2;
        if (layerdisk_ == LayerDisk::D1)
          iseed = Seed::L2L3D1;
        inner = 2;
      } else {
        throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";
      }
      assert(iseed != -1);
      int seedindex = -1;
      for (unsigned int k = 0; k < vmstubsTEPHI_.size(); k++) {
        if (vmstubsTEPHI_[k].seednumber == (unsigned int)iseed) {
          seedindex = k;
        }
      }
      if (seedindex == -1) {
        seedindex = vmstubsTEPHI_.size();
        vector<VMStubsTEMemory*> avectmp;
        vector<vector<VMStubsTEMemory*> > vectmp(settings_.nvmte(inner, iseed), avectmp);
        VMStubsTEPHI atmp(iseed, inner, vectmp);
        vmstubsTEPHI_.push_back(atmp);
      }
      vmstubsTEPHI_[seedindex].vmstubmem[(vmbin - 1) & (settings_.nvmte(inner, iseed) - 1)].push_back(tmp);

    } else if (memory->getName().substr(3, 2) == "ME") {
      VMStubsMEMemory* tmp = dynamic_cast<VMStubsMEMemory*>(memory);
      assert(tmp != nullptr);
      vmstubsMEPHI_[(vmbin - 1) & (settings_.nvmme(layerdisk_) - 1)] = tmp;
    } else {
      throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " should never get here!";
    }

    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void VMRouter::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "stubin") {
    InputLinkMemory* tmp1 = dynamic_cast<InputLinkMemory*>(memory);
    assert(tmp1 != nullptr);
    if (tmp1 != nullptr) {
      if (layerdisk_ >= N_LAYER && tmp1->getName().find("2S_") != string::npos) {
        stubinputdisk2stmp_.push_back(tmp1);
      } else {
        stubinputtmp_.push_back(tmp1);
      }
    }
    //This gymnastic is done to ensure that in the disks the PS stubs are processed before
    //the 2S stubs. This is needed by the current HLS implemenation of the VM router.
    stubinputs_ = stubinputtmp_;
    for (auto& mem : stubinputdisk2stmp_) {
      stubinputs_.push_back(mem);
    }
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void VMRouter::execute() {
  unsigned int allStubCounter = 0;

  //Loop over the input stubs
  for (auto& stubinput : stubinputs_) {
    for (unsigned int i = 0; i < stubinput->nStubs(); i++) {
      if (allStubCounter >= settings_.maxStep("VMR"))
        continue;
      if (allStubCounter >= (1 << N_BITSMEMADDRESS))
        continue;
      Stub* stub = stubinput->getStub(i);

      //Note - below information is not part of the stub, but rather from which input memory we are reading
      bool negdisk = (stub->disk().value() < 0);

      //use &127 to make sure we fit into the number of bits -
      //though we should have protected against overflows above
      FPGAWord allStubIndex(allStubCounter & ((1 << N_BITSMEMADDRESS) - 1), N_BITSMEMADDRESS, true, __LINE__, __FILE__);

      //TODO - should not be needed - but need to migrate some other pieces of code before removing
      stub->setAllStubIndex(allStubCounter);
      //TODO - should not be needed - but need to migrate some other pieces of code before removing
      stub->l1tstub()->setAllStubIndex(allStubCounter);

      allStubCounter++;

      //Fill allstubs memories - in HLS this is the same write to multiple memories
      for (auto& allstub : allstubs_) {
        allstub->addStub(stub);
      }

      //Fill all the ME VM memories

      FPGAWord iphi = stub->phicorr();
      unsigned int ivm =
          iphi.bits(iphi.nbits() - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_)),
                    settings_.nbitsvmme(layerdisk_));
      unsigned int extrabits = iphi.bits(iphi.nbits() - overlapbits_, nextrabits_);

      unsigned int ivmPlus = ivm;

      if (extrabits == ((1U << nextrabits_) - 1) && ivm != ((1U << settings_.nbitsvmme(layerdisk_)) - 1))
        ivmPlus++;
      unsigned int ivmMinus = ivm;
      if (extrabits == 0 && ivm != 0)
        ivmMinus--;

      //Calculate the z and r position for the vmstub

      //Take the top nbitszfinebintable_ bits of the z coordinate
      int indexz = (((1 << (stub->z().nbits() - 1)) + stub->z().value()) >> (stub->z().nbits() - nbitszfinebintable_));
      int indexr = -1;
      if (layerdisk_ > (N_LAYER - 1)) {
        if (negdisk) {
          indexz = (1 << nbitszfinebintable_) - indexz;
        }
        indexr = stub->r().value();
        if (stub->isPSmodule()) {
          indexr = stub->r().value() >> (stub->r().nbits() - nbitsrfinebintable_);
        }
      } else {
        //Take the top nbitsfinebintable_ bits of the z coordinate. The & is to handle the negative z values.
        indexr = (((1 << (stub->r().nbits() - 1)) + stub->r().value()) >> (stub->r().nbits() - nbitsrfinebintable_));
      }

      assert(indexz >= 0);
      assert(indexr >= 0);
      assert(indexz < (1 << nbitszfinebintable_));
      assert(indexr < (1 << nbitsrfinebintable_));

      int melut = meTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      assert(melut >= 0);

      int vmbin = melut >> 3;
      if (negdisk)
        vmbin += 8;
      int rzfine = melut & 7;

      // pad disk PS bend word with a '0' in MSB so that all disk bends have 4 bits (for HLS compatibility)
      int nbendbits = stub->bend().nbits();
      if (layerdisk_ >= N_LAYER)
        nbendbits = settings_.nbendbitsmedisk();

      VMStubME vmstub(stub,
                      stub->iphivmFineBins(settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_), 3),
                      FPGAWord(rzfine, 3, true, __LINE__, __FILE__),
                      FPGAWord(stub->bend().value(), nbendbits, true, __LINE__, __FILE__),
                      allStubIndex);

      assert(vmstubsMEPHI_[ivmPlus] != nullptr);
      vmstubsMEPHI_[ivmPlus]->addStub(vmstub, vmbin);
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << getName() << " adding stub to " << vmstubsMEPHI_[ivmPlus]->getName()
                                     << " ivmPlus" << ivmPlus << " bin=" << vmbin;
      }

      if (ivmMinus != ivmPlus) {
        assert(vmstubsMEPHI_[ivmMinus] != nullptr);
        vmstubsMEPHI_[ivmMinus]->addStub(vmstub, vmbin);
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " adding stub to " << vmstubsMEPHI_[ivmMinus]->getName()
                                       << " ivmMinus" << ivmMinus << " bin=" << vmbin;
        }
      }

      //Fill the TE VM memories

      for (auto& ivmstubTEPHI : vmstubsTEPHI_) {
        unsigned int iseed = ivmstubTEPHI.seednumber;
        unsigned int inner = ivmstubTEPHI.stubposition;
        if ((iseed == Seed::D1D2 || iseed == Seed::D3D4 || iseed == Seed::L1D1 || iseed == Seed::L2D1) &&
            (!stub->isPSmodule()))
          continue;

        unsigned int lutwidth = settings_.lutwidthtab(inner, iseed);
        if (settings_.extended()) {
          lutwidth = settings_.lutwidthtabextended(inner, iseed);
        }

        int lutval = -999;

        if (inner > 0) {
          if (layerdisk_ < N_LAYER) {
            lutval = melut;
          } else {
            if (inner == 2 && iseed == Seed::L2L3D1) {
              lutval = 0;
              if (stub->r().value() < 10) {
                lutval = 8 * (1 + (stub->r().value() >> 2));
              } else {
                if (stub->r().value() < settings_.rmindiskl3overlapvm() / settings_.kr()) {
                  lutval = -1;
                }
              }
            } else {
              lutval = diskTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
            }
          }
          if (lutval == -1)
            continue;
        } else {
          if (iseed < Seed::L1D1 || iseed > Seed::L2D1) {
            lutval = innerTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
          } else {
            lutval = innerOverlapTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
          }
          if (lutval == -1)
            continue;
          if (settings_.extended() &&
              (iseed == Seed::L3L4 || iseed == Seed::L5L6 || iseed == Seed::D1D2 || iseed == Seed::L2L3D1)) {
            int lutval2 = innerThirdTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
            if (lutval2 != -1)
              lutval += (lutval2 << 10);
          }
        }

        assert(lutval >= 0);

        FPGAWord binlookup(lutval, lutwidth, true, __LINE__, __FILE__);

        if (binlookup.value() < 0)
          continue;

        unsigned int ivmte =
            iphi.bits(iphi.nbits() - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmte(inner, iseed)),
                      settings_.nbitsvmte(inner, iseed));

        int bin = -1;
        if (inner != 0) {
          bin = binlookup.value() / 8;
          unsigned int tmp = binlookup.value() & 7;  //three bits in outer layers - this could be coded cleaner...
          binlookup.set(tmp, 3, true, __LINE__, __FILE__);
        }

        FPGAWord finephi = stub->iphivmFineBins(settings_.nphireg(inner, iseed), settings_.nfinephi(inner, iseed));

        VMStubTE tmpstub(stub, finephi, stub->bend(), binlookup, allStubIndex);

        unsigned int nmem = ivmstubTEPHI.vmstubmem[ivmte].size();

        assert(nmem > 0);

        for (unsigned int l = 0; l < nmem; l++) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << getName() << " try adding stub to "
                                         << ivmstubTEPHI.vmstubmem[ivmte][l]->getName() << " inner=" << inner
                                         << " bin=" << bin;
          }
          if (inner == 0) {
            ivmstubTEPHI.vmstubmem[ivmte][l]->addVMStub(tmpstub);
          } else {
            ivmstubTEPHI.vmstubmem[ivmte][l]->addVMStub(tmpstub, bin);
          }
        }
      }
    }
  }
}

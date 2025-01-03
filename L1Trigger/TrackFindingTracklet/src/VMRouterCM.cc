#include "L1Trigger/TrackFindingTracklet/interface/VMRouterCM.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"
#include "L1Trigger/TrackFindingTracklet/interface/InputLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllInnerStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

VMRouterCM::VMRouterCM(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global),
      meTable_(settings),
      diskTable_(settings),
      meTableOld_(settings),
      diskTableOld_(settings),
      innerTable_(settings),
      innerOverlapTable_(settings),
      innerThirdTable_(settings) {
  layerdisk_ = initLayerDisk(4);

  unsigned int region = name[9] - 'A';
  assert(region < settings_.nallstubs(layerdisk_));

  overlapbits_ = 7;
  nextrabits_ = overlapbits_ - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_));

  // The TrackletProcessorDisplaced currently uses the older LUTs that were
  // used with the non-combined modules. To maintain compatibility, we
  // initialize these older LUTs below, which are used for the triplet seeds in
  // the "execute" method. Once the TrackletProcessorDisplaced is updated,
  // these can be removed.

  meTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::me, region);            //used for ME and outer TE barrel
  meTableOld_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::me, region, false);  //used for ME and outer TE barrel

  if (layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D2 || layerdisk_ == LayerDisk::D4) {
    diskTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::disk, region);  //outer disk used by D1, D2, and D4
    diskTableOld_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::disk, region, false);  //outer disk used by D1, D2, and D4
  }

  if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3 ||
      layerdisk_ == LayerDisk::L5 || layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3) {
    innerTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::inner, region, false);  //projection to next layer/disk
  }

  if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2) {
    innerOverlapTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::inneroverlap, region, false);  //projection to disk from layer
  }

  if (layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L5 ||
      layerdisk_ == LayerDisk::D1) {
    innerThirdTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::innerthird, region, false);  //projection to third layer/disk
  }

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk_);
}

void VMRouterCM::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }

  if (output == "allinnerstubout") {
    AllInnerStubsMemory* tmp = dynamic_cast<AllInnerStubsMemory*>(memory);
    assert(tmp != nullptr);
    char memtype = memory->getName().back();
    allinnerstubs_.emplace_back(memtype, tmp);
    return;
  }

  if (output.substr(0, 10) == "allstubout") {
    AllStubsMemory* tmp = dynamic_cast<AllStubsMemory*>(memory);
    allstubs_.push_back(tmp);
    return;
  }

  if (output.substr(0, 9) == "vmstubout") {
    if (memory->getName().substr(3, 2) == "TE") {
      VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
      int i = output.find_last_of('_');
      unsigned int iseed = std::stoi(output.substr(i + 1));
      assert(iseed < N_SEED);

      // This flag is used to replicate the behavior of the old VMRouter for
      // the case of the triplet seeds.
      const bool isTripletSeed = (iseed >= L2L3L4);

      // seedtype, vmbin, and inner are only used in the case of the triplet
      // seeds.
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
      unsigned int inner = 1;
      if (seedtype < 'I') {
        if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L5 ||
            layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3)
          inner = 0;
      } else if (seedtype < 'M') {
        if (layerdisk_ == LayerDisk::L2)
          inner = 0;
      } else if (seedtype <= 'Z') {
        if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2)
          inner = 0;
      } else if (seedtype < 'o' && seedtype >= 'a') {
        if (layerdisk_ == LayerDisk::L2)
          inner = 0;
      } else if (seedtype > 'o' && seedtype <= 'z') {
        inner = 2;
      } else {
        throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";
      }

      int seedindex = -1;
      for (unsigned int k = 0; k < vmstubsTEPHI_.size(); k++) {
        if (vmstubsTEPHI_[k].seednumber == iseed) {
          seedindex = k;
        }
      }
      if (seedindex == -1) {
        seedindex = vmstubsTEPHI_.size();
        vector<VMStubsTEMemory*> avectmp;
        vector<vector<VMStubsTEMemory*> > vectmp(!isTripletSeed ? 1 : settings_.nvmte(inner, iseed), avectmp);
        VMStubsTEPHICM atmp(iseed, inner, vectmp);
        vmstubsTEPHI_.push_back(atmp);
      }

      if (!isTripletSeed) {
        tmp->resize(settings_.NLONGVMBINS() * settings_.nvmte(1, iseed));
        vmstubsTEPHI_[seedindex].vmstubmem[0].push_back(tmp);
      } else {
        vmstubsTEPHI_[seedindex].vmstubmem[(vmbin - 1) & (settings_.nvmte(inner, iseed) - 1)].push_back(tmp);
      }
    } else {
      throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " memory: " << memory->getName()
                                         << " => should never get here!";
    }

    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void VMRouterCM::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "stubin") {
    InputLinkMemory* tmp1 = dynamic_cast<InputLinkMemory*>(memory);
    assert(tmp1 != nullptr);
    if (tmp1 != nullptr) {
      stubinputs_.push_back(tmp1);
    }
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void VMRouterCM::execute(unsigned int) {
  unsigned int allStubCounter = 0;

  //bool print = getName() == "VMR_D1PHIB" && iSector == 3;
  //print = false;

  //Loop over the input stubs
  for (auto& stubinput : stubinputs_) {
    for (unsigned int i = 0; i < stubinput->nStubs(); i++) {
      if (allStubCounter >= settings_.maxStep("VMR"))
        continue;
      if (allStubCounter >= (1 << N_BITSMEMADDRESS))
        continue;

      Stub* stub = stubinput->getStub(i);

      //Note - below information is not part of the stub, but rather from which input memory
      //we are reading
      bool negdisk = (stub->disk().value() < 0);

      //use &127 to make sure we fit into the number of bits -
      //though we should have protected against overflows above
      FPGAWord allStubIndex(allStubCounter & ((1 << N_BITSMEMADDRESS) - 1), N_BITSMEMADDRESS, true, __LINE__, __FILE__);

      //TODO - should not be needed - but need to migrate some other pieces of code before removing
      stub->setAllStubIndex(allStubCounter);
      //TODO - should not be needed - but need to migrate some other pieces of code before removing
      stub->l1tstub()->setAllStubIndex(allStubCounter);

      allStubCounter++;

      for (auto& allstub : allstubs_) {
        allstub->addStub(stub);
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " adding stub to " << allstub->getName();
        }
      }

      FPGAWord iphi = stub->phicorr();
      unsigned int iphipos = iphi.bits(iphi.nbits() - (settings_.nbitsallstubs(layerdisk_) + N_PHIBITS), N_PHIBITS);

      unsigned int phicutmax = 4;
      unsigned int phicutmin = 4;

      if (layerdisk_ != 0) {
        phicutmax = 6;
        phicutmin = 2;
      }

      //Fill inner allstubs memories - in HLS this is the same write to multiple memories
      for (auto& allstub : allinnerstubs_) {
        char memtype = allstub.first;
        if (memtype == 'R' && iphipos < phicutmax)
          continue;
        if (memtype == 'L' && iphipos >= phicutmin)
          continue;
        if (memtype == 'A' && iphipos < 4)
          continue;
        if (memtype == 'B' && iphipos >= 4)
          continue;
        if (memtype == 'E' && iphipos >= 4)
          continue;
        if (memtype == 'F' && iphipos < 4)
          continue;
        if (memtype == 'C' && iphipos >= 4)
          continue;
        if (memtype == 'D' && iphipos < 4)
          continue;

        int absz = std::abs(stub->z().value());
        if (layerdisk_ == LayerDisk::L2 && absz < VMROUTERCUTZL2 / settings_.kz(layerdisk_))
          continue;
        if ((layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L5) &&
            absz > VMROUTERCUTZL1L3L5 / settings_.kz(layerdisk_))
          continue;
        if ((layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3) &&
            stub->rvalue() > VMROUTERCUTRD1D3 / settings_.kr())
          continue;
        if ((layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3) && stub->rvalue() < 2 * int(N_DSS_MOD))
          continue;
        if (layerdisk_ == LayerDisk::L1) {
          if (memtype == 'M' || memtype == 'R' || memtype == 'L') {
            if (absz < VMROUTERCUTZL1 / settings_.kz(layerdisk_))
              continue;
          } else {
            if (absz > VMROUTERCUTZL1L3L5 / settings_.kz(layerdisk_))
              continue;
          }
        }

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " adding stub to " << allstub.second->getName();
        }

        allstub.second->addStub(stub);
      }

      //Calculate the z and r position for the vmstub
      //Take the top nbitszfinebintable_ bits of the z coordinate
      int indexz = (stub->z().value() >> (stub->z().nbits() - nbitszfinebintable_)) & ((1 << nbitszfinebintable_) - 1);
      int indexr = -1;
      if (layerdisk_ > (N_LAYER - 1)) {
        if (negdisk) {
          indexz = ((1 << nbitszfinebintable_) - 1) - indexz;
        }
        indexr = stub->rvalue();
        if (stub->isPSmodule()) {
          indexr = stub->rvalue() >> (stub->r().nbits() + 1 - nbitsrfinebintable_);
        }
      } else {
        //Take the top nbitsfinebintable_ bits of the z coordinate. The & is to handle the negative z values.
        indexr = (stub->rvalue() >> (stub->r().nbits() - nbitsrfinebintable_)) & ((1 << nbitsrfinebintable_) - 1);
      }

      assert(indexz >= 0);
      assert(indexr >= 0);
      assert(indexz < (1 << nbitszfinebintable_));
      assert(indexr < (1 << nbitsrfinebintable_));

      int melut = meTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      assert(melut >= 0);

      // The following indices are calculated in the same way as in the old
      // VMRouter and are only used for the triplet seeds.
      int indexzOld =
          (((1 << (stub->z().nbits() - 1)) + stub->z().value()) >> (stub->z().nbits() - nbitszfinebintable_));
      int indexrOld = -1;
      if (layerdisk_ > (N_LAYER - 1)) {
        if (negdisk) {
          indexzOld = (1 << nbitszfinebintable_) - indexzOld;
        }
        indexrOld = stub->r().value();
        if (stub->isPSmodule()) {
          indexrOld = stub->r().value() >> (stub->r().nbits() - nbitsrfinebintable_);
        }
      } else {
        //Take the top nbitsfinebintable_ bits of the z coordinate. The & is to handle the negative z values.
        indexrOld = (((1 << (stub->r().nbits() - 1)) + stub->r().value()) >> (stub->r().nbits() - nbitsrfinebintable_));
      }

      assert(indexzOld >= 0);
      assert(indexrOld >= 0);
      assert(indexzOld < (1 << nbitszfinebintable_));
      assert(indexrOld < (1 << nbitsrfinebintable_));

      int melutOld = meTableOld_.lookup((indexzOld << nbitsrfinebintable_) + indexrOld);

      assert(melutOld >= 0);

      //Fill the TE VM memories
      if (layerdisk_ >= N_LAYER && (!stub->isPSmodule()))
        continue;

      for (auto& ivmstubTEPHI : vmstubsTEPHI_) {
        unsigned int iseed = ivmstubTEPHI.seednumber;

        // This flag is used to replicate the behavior of the old VMRouter for
        // the case of the triplet seeds.
        const bool isTripletSeed = (iseed >= L2L3L4);

        if (!isTripletSeed && layerdisk_ >= N_LAYER && (!stub->isPSmodule()))
          continue;
        unsigned int inner = (!isTripletSeed ? 1 : ivmstubTEPHI.stubposition);
        unsigned int lutwidth = settings_.lutwidthtab(inner, iseed);
        if (settings_.extended()) {
          lutwidth = settings_.lutwidthtabextended(inner, iseed);
        }

        int lutval = -999;

        if (inner > 0) {
          if (layerdisk_ < N_LAYER) {
            lutval = (!isTripletSeed ? melut : melutOld);
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
              lutval = (!isTripletSeed ? diskTable_.lookup((indexz << nbitsrfinebintable_) + indexr)
                                       : diskTableOld_.lookup((indexzOld << nbitsrfinebintable_) + indexrOld));
              if (lutval == 0)
                continue;
            }
          }
          if (lutval == -1)
            continue;
        } else {
          if (iseed < Seed::L1D1 || iseed > Seed::L2D1) {
            lutval = innerTable_.lookup((indexzOld << nbitsrfinebintable_) + indexrOld);
          } else {
            lutval = innerOverlapTable_.lookup((indexzOld << nbitsrfinebintable_) + indexrOld);
          }
          if (lutval == -1)
            continue;
          if (settings_.extended() &&
              (iseed == Seed::L3L4 || iseed == Seed::L5L6 || iseed == Seed::D1D2 || iseed == Seed::L2L3D1)) {
            int lutval2 = innerThirdTable_.lookup((indexzOld << nbitsrfinebintable_) + indexrOld);
            if (lutval2 != -1) {
              const auto& lutshift = innerTable_.nbits();  // should be same for all inner tables
              lutval += (lutval2 << lutshift);
            }
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
          bin = binlookup.value() >> settings_.NLONGVMBITS();
          unsigned int tmp = binlookup.value() & (settings_.NLONGVMBINS() - 1);  //three bits in outer layers
          binlookup.set(tmp, settings_.NLONGVMBITS(), true, __LINE__, __FILE__);
        }

        FPGAWord finephi = stub->iphivmFineBins(settings_.nphireg(inner, iseed), settings_.nfinephi(inner, iseed));

        VMStubTE tmpstub(stub, finephi, stub->bend(), binlookup, allStubIndex);

        unsigned int nmem = ivmstubTEPHI.vmstubmem[!isTripletSeed ? 0 : ivmte].size();
        assert(nmem > 0);

        for (unsigned int l = 0; l < nmem; l++) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << getName() << " try adding stub to "
                                         << ivmstubTEPHI.vmstubmem[!isTripletSeed ? 0 : ivmte][l]->getName()
                                         << " bin=" << bin << " ivmte " << ivmte << " finephi " << finephi.value()
                                         << " regions bits " << settings_.nphireg(1, iseed) << " finephibits "
                                         << settings_.nfinephi(1, iseed);
          }
          if (!isTripletSeed)
            ivmstubTEPHI.vmstubmem[0][l]->addVMStub(tmpstub, bin, ivmte);
          else {
            if (inner == 0) {
              ivmstubTEPHI.vmstubmem[ivmte][l]->addVMStub(tmpstub);
            } else {
              ivmstubTEPHI.vmstubmem[ivmte][l]->addVMStub(tmpstub, bin, 0, false);
            }
          }
        }
      }
    }
  }
}

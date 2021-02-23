#include "L1Trigger/TrackFindingTracklet/interface/VMRouterCM.h"
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

VMRouterCM::VMRouterCM(string name, Settings const& settings, Globals* global, unsigned int iSector)
    : ProcessBase(name, settings, global, iSector), vmrtable_(settings) {
  layerdisk_ = initLayerDisk(4);

  vmstubsMEPHI_.resize(1, nullptr);

  overlapbits_ = 7;
  nextrabits_ = overlapbits_ - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_));

  vmrtable_.init(layerdisk_, getName());

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk_);

  nvmmebins_ = settings_.NLONGVMBINS() * ((layerdisk_ >= 6) ? 2 : 1);  //number of long z/r bins in VM
}

void VMRouterCM::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }

  if (output.substr(0, 10) == "allstubout") {
    AllStubsMemory* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    char memtype = 0;
    if (output.size() > 10) {
      memtype = output[11];
    }
    allstubs_.emplace_back(memtype, tmp);
    return;
  }

  if (output.substr(0, 9) == "vmstubout") {
    unsigned int pos = 12;
    int vmbin = memory->getName().substr(pos, 1)[0] - '0';
    pos++;
    if (pos < memory->getName().size()) {
      if (memory->getName().substr(pos, 1)[0] != 'n') {
        vmbin = vmbin * 10 + memory->getName().substr(pos, 1)[0] - '0';
        pos++;
      }
    }
    if (memory->getName().substr(3, 2) == "TE") {
      VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
      int iseed = output[output.size() - 1] - '0';
      assert(iseed >= 0);
      assert(iseed < 8);

      int seedindex = -1;
      for (unsigned int k = 0; k < vmstubsTEPHI_.size(); k++) {
        if (vmstubsTEPHI_[k].seednumber == (unsigned int)iseed) {
          seedindex = k;
        }
      }
      if (seedindex == -1) {
        seedindex = vmstubsTEPHI_.size();
        vector<VMStubsTEMemory*> vectmp;
        VMStubsTEPHICM atmp(iseed, vectmp);
        vmstubsTEPHI_.push_back(atmp);
      }
      tmp->resize(settings_.NLONGVMBINS() * settings_.nvmte(1, iseed));
      vmstubsTEPHI_[seedindex].vmstubmem.push_back(tmp);

    } else if (memory->getName().substr(3, 2) == "ME") {
      VMStubsMEMemory* tmp = dynamic_cast<VMStubsMEMemory*>(memory);
      assert(tmp != nullptr);
      tmp->resize(nvmmebins_ * settings_.nvmme(layerdisk_));
      assert(vmstubsMEPHI_[0] == nullptr);
      vmstubsMEPHI_[0] = tmp;
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

void VMRouterCM::execute() {
  unsigned int allStubCounter = 0;

  //Loop over the input stubs
  for (auto& stubinput : stubinputs_) {
    for (unsigned int i = 0; i < stubinput->nStubs(); i++) {
      if (allStubCounter > settings_.maxStep("VMR"))
        continue;
      if (allStubCounter > 127)
        continue;

      Stub* stub = stubinput->getStub(i);

      //Note - below information is not part of the stub, but rather from which input memory
      //we are reading
      bool negdisk = (stub->disk().value() < 0);

      //use &127 to make sure we fit into the number of bits -
      //though we should have protected against overflows above
      FPGAWord allStubIndex(allStubCounter & 127, 7, true, __LINE__, __FILE__);

      //TODO - should not be needed - but need to migrate some other pieces of code before removing
      stub->setAllStubIndex(allStubCounter);
      //TODO - should not be needed - but need to migrate some other pieces of code before removing
      stub->l1tstub()->setAllStubIndex(allStubCounter);

      allStubCounter++;

      FPGAWord iphi = stub->phicorr();
      unsigned int iphipos = iphi.bits(iphi.nbits() - (settings_.nbitsallstubs(layerdisk_) + 3), 3);

      //Fill allstubs memories - in HLS this is the same write to multiple memories
      for (auto& allstub : allstubs_) {
        char memtype = allstub.first;
        if ((memtype == 'R' && iphipos < 5) || (memtype == 'L' && iphipos >= 3) || (memtype == 'A' && iphipos < 4) ||
            (memtype == 'B' && iphipos >= 4) || (memtype == 'E' && iphipos >= 4) || (memtype == 'F' && iphipos < 4) ||
            (memtype == 'C' && iphipos >= 4) || (memtype == 'D' && iphipos < 4))
          continue;
        allstub.second->addStub(stub);
      }

      //Fill all the ME VM memories
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

      int melut = vmrtable_.lookup(indexz, indexr);

      assert(melut >= 0);

      int vmbin = melut >> NFINERZBITS;
      if (negdisk)
        vmbin += (1 << NFINERZBITS);
      int rzfine = melut & ((1 << NFINERZBITS) - 1);

      // pad disk PS bend word with a '0' in MSB so that all disk bends have 4 bits (for HLS compatibility)
      int nbendbits = stub->bend().nbits();
      if (layerdisk_ >= N_LAYER)
        nbendbits = settings_.nbendbitsmedisk();

      VMStubME vmstub(
          stub,
          stub->iphivmFineBins(settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_), NFINERZBITS),
          FPGAWord(rzfine, NFINERZBITS, true, __LINE__, __FILE__),
          FPGAWord(stub->bend().value(), nbendbits, true, __LINE__, __FILE__),
          allStubIndex);

      assert(vmstubsMEPHI_[0] != nullptr);
      vmstubsMEPHI_[0]->addStub(vmstub, ivmPlus * nvmmebins_ + vmbin);

      if (ivmMinus != ivmPlus) {
        vmstubsMEPHI_[0]->addStub(vmstub, ivmMinus * nvmmebins_ + vmbin);
      }

      //Fill the TE VM memories
      if (layerdisk_ >= 6 && (!stub->isPSmodule()))
        continue;

      for (auto& ivmstubTEPHI : vmstubsTEPHI_) {
        unsigned int iseed = ivmstubTEPHI.seednumber;
        unsigned int lutwidth = settings_.lutwidthtab(1, iseed);

        int lutval = -999;

        if (layerdisk_ < N_LAYER) {
          lutval = melut;
        } else {
          lutval = vmrtable_.lookupdisk(indexz, indexr);
        }
        if (lutval == -1)
          continue;

        assert(lutval >= 0);

        FPGAWord binlookup(lutval, lutwidth, true, __LINE__, __FILE__);

        if (binlookup.value() < 0)
          continue;

        unsigned int ivmte =
            iphi.bits(iphi.nbits() - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmte(1, iseed)),
                      settings_.nbitsvmte(1, iseed));

        int bin = binlookup.value() / 8;
        unsigned int tmp = binlookup.value() & 7;  //three bits in outer layers - this could be coded cleaner...
        binlookup.set(tmp, 3, true, __LINE__, __FILE__);

        FPGAWord finephi = stub->iphivmFineBins(settings_.nphireg(1, iseed), settings_.nfinephi(1, iseed));

        VMStubTE tmpstub(stub, finephi, stub->bend(), binlookup, allStubIndex);

        unsigned int nmem = ivmstubTEPHI.vmstubmem.size();
        assert(nmem > 0);

        for (unsigned int l = 0; l < nmem; l++) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << getName() << " try adding stub to " << ivmstubTEPHI.vmstubmem[l]->getName()
                                         << " bin=" << bin << " ivmte " << ivmte << " finephi " << finephi.value()
                                         << " regions bits " << settings_.nphireg(1, iseed) << " finephibits "
                                         << settings_.nfinephi(1, iseed);
          }
          ivmstubTEPHI.vmstubmem[l]->addVMStub(tmpstub, ivmte * settings_.NLONGVMBINS() + bin);
        }
      }
    }
  }
}

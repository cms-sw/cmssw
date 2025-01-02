#include "L1Trigger/TrackFindingTracklet/interface/VMStubMERouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

VMStubMERouter::VMStubMERouter(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global), meTable_(settings) {
  layerdisk_ = initLayerDisk(7);

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk_);

  unsigned int region = name[12] - 'A';
  assert(region < settings_.nallstubs(layerdisk_));

  meTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::me, region);  //used for ME and outer TE barrel

  vmstubsMEPHI_.resize(1, nullptr);
  nvmmebins_ = settings_.NLONGVMBINS() * ((layerdisk_ >= N_LAYER) ? 2 : 1);
}

void VMStubMERouter::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }

  if (output.substr(0, 10) == "allstubout") {
    AllStubsMemory* tmp = dynamic_cast<AllStubsMemory*>(memory);
    allstubs_.push_back(tmp);
    return;
  }

  if (output == "vmstubout") {
    VMStubsMEMemory* tmp = dynamic_cast<VMStubsMEMemory*>(memory);
    assert(tmp != nullptr);
    tmp->resize(16 * settings_.nvmme(layerdisk_));
    assert(vmstubsMEPHI_.size()<=2);
    if (vmstubsMEPHI_[0] == nullptr) {
      vmstubsMEPHI_[0] = tmp;
    } else {
      vmstubsMEPHI_.push_back(tmp);
    }
    
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void VMStubMERouter::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "allstubin") {
    AllStubsMemory* tmp1 = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp1 != nullptr);
    if (tmp1 != nullptr) {
      stubinputs_.push_back(tmp1);
    }
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void VMStubMERouter::execute(unsigned int) {
  unsigned int allStubCounter = 0;

  //Loop over the input stubs
  for (auto& stubinput : stubinputs_) {
    for (unsigned int i = 0; i < stubinput->nStubs(); i++) {
      if (allStubCounter > settings_.maxStep("VMR"))
        continue;
      if (allStubCounter >= (1 << N_BITSMEMADDRESS))
        continue;

      FPGAWord allStubIndex(allStubCounter & ((1 << N_BITSMEMADDRESS) - 1), N_BITSMEMADDRESS, true, __LINE__, __FILE__);
      const Stub* stub = stubinput->getStub(i);
      allStubCounter++;

      for (auto& allstub : allstubs_) {
        allstub->addStub(stub);
      }

      FPGAWord iphi = stub->phicorr();

      bool negdisk = (stub->disk().value() < 0);
      //Fill all the ME VM memories
      unsigned int ivm =
          iphi.bits(iphi.nbits() - (settings_.nbitsallstubs(layerdisk_) + settings_.nbitsvmme(layerdisk_)),
                    settings_.nbitsvmme(layerdisk_));

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

      if (vmstubsMEPHI_[0] != nullptr) {
	for (unsigned int i=0; i<vmstubsMEPHI_.size(); i++) {
	  vmstubsMEPHI_[i]->addStub(vmstub, ivm * nvmmebins_ + vmbin);
	}
      }
    }
  }
}


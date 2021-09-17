#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <filesystem>

using namespace trklet;
using namespace std;

TrackletEngine::TrackletEngine(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global), innerptlut_(settings), outerptlut_(settings) {
  stubpairs_ = nullptr;
  innervmstubs_ = nullptr;
  outervmstubs_ = nullptr;

  initLayerDisksandISeed(layerdisk1_, layerdisk2_, iSeed_);

  innerphibits_ = settings.nfinephi(0, iSeed_);
  outerphibits_ = settings.nfinephi(1, iSeed_);
}

void TrackletEngine::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "stubpairout") {
    StubPairsMemory* tmp = dynamic_cast<StubPairsMemory*>(memory);
    assert(tmp != nullptr);
    stubpairs_ = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void TrackletEngine::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "innervmstubin") {
    VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    innervmstubs_ = tmp;
    setVMPhiBin();
    return;
  }
  if (input == "outervmstubin") {
    VMStubsTEMemory* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    outervmstubs_ = tmp;
    setVMPhiBin();
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TrackletEngine::execute() {
  if (!settings_.useSeed(iSeed_))
    return;

  unsigned int countall = 0;
  unsigned int countpass = 0;

  assert(innervmstubs_ != nullptr);
  assert(outervmstubs_ != nullptr);

  for (unsigned int i = 0; i < innervmstubs_->nVMStubs(); i++) {
    const VMStubTE& innervmstub = innervmstubs_->getVMStubTE(i);
    FPGAWord lookupbits = innervmstub.vmbits();

    unsigned int nbits = 7;
    if (iSeed_ == 4 || iSeed_ == 5)
      nbits = 6;
    int rzdiffmax = lookupbits.bits(nbits, lookupbits.nbits() - nbits);
    int rzbinfirst = lookupbits.bits(0, 3);
    int start = lookupbits.bits(4, nbits - 4);
    int next = lookupbits.bits(3, 1);

    if ((iSeed_ == 4 || iSeed_ == 5) && innervmstub.stub()->disk().value() < 0) {  //TODO - need to store negative disk
      start += 4;
    }
    int last = start + next;

    for (int ibin = start; ibin <= last; ibin++) {
      for (unsigned int j = 0; j < outervmstubs_->nVMStubsBinned(ibin); j++) {
        if (countall >= settings_.maxStep("TE"))
          break;
        countall++;
        const VMStubTE& outervmstub = outervmstubs_->getVMStubTEBinned(ibin, j);

        int rzbin = outervmstub.vmbits().bits(0, 3);

        FPGAWord iphiinnerbin = innervmstub.finephi();
        FPGAWord iphiouterbin = outervmstub.finephi();

        unsigned int index = (iphiinnerbin.value() << outerphibits_) + iphiouterbin.value();

        if (iSeed_ >= 4) {  //Also use r-position
          int ir = ((ibin & 3) << 1) + (rzbin >> 2);
          index = (index << 3) + ir;
        }

        if (start != ibin)
          rzbin += 8;
        if ((rzbin < rzbinfirst) || (rzbin - rzbinfirst > rzdiffmax)) {
          continue;
        }

        FPGAWord innerbend = innervmstub.bend();
        FPGAWord outerbend = outervmstub.bend();

        int ptinnerindex = (index << innerbend.nbits()) + innerbend.value();
        int ptouterindex = (index << outerbend.nbits()) + outerbend.value();

        if (!(innerptlut_.lookup(ptinnerindex) && outerptlut_.lookup(ptouterindex))) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << "Stub pair rejected because of stub pt cut bends : "
                                         << settings_.benddecode(
                                                innervmstub.bend().value(), layerdisk1_, innervmstub.isPSmodule())
                                         << " "
                                         << settings_.benddecode(
                                                outervmstub.bend().value(), layerdisk2_, outervmstub.isPSmodule());
          }
          continue;
        }

        if (settings_.debugTracklet())
          edm::LogVerbatim("Tracklet") << "Adding stub pair in " << getName();

        stubpairs_->addStubPair(innervmstub, outervmstub);
        countpass++;
      }
    }
  }

  if (settings_.writeMonitorData("TE")) {
    globals_->ofstream("trackletengine.txt") << getName() << " " << countall << " " << countpass << endl;
  }
}

void TrackletEngine::setVMPhiBin() {
  if (innervmstubs_ == nullptr || outervmstubs_ == nullptr)
    return;

  innervmstubs_->setother(outervmstubs_);
  outervmstubs_->setother(innervmstubs_);

  double innerphimin, innerphimax;
  innervmstubs_->getPhiRange(innerphimin, innerphimax, iSeed_, 0);

  double outerphimin, outerphimax;
  outervmstubs_->getPhiRange(outerphimin, outerphimax, iSeed_, 1);

  string innermem = innervmstubs_->getName().substr(6);
  string outermem = outervmstubs_->getName().substr(6);

  innerptlut_.initteptlut(true,
                          false,
                          iSeed_,
                          layerdisk1_,
                          layerdisk2_,
                          innerphibits_,
                          outerphibits_,
                          innerphimin,
                          innerphimax,
                          outerphimin,
                          outerphimax,
                          innermem,
                          outermem);

  outerptlut_.initteptlut(false,
                          false,
                          iSeed_,
                          layerdisk1_,
                          layerdisk2_,
                          innerphibits_,
                          outerphibits_,
                          innerphimin,
                          innerphimax,
                          outerphimin,
                          outerphimax,
                          innermem,
                          outermem);

  TrackletLUT innertememlut(settings_);
  TrackletLUT outertememlut(settings_);

  innertememlut.initteptlut(true,
                            true,
                            iSeed_,
                            layerdisk1_,
                            layerdisk2_,
                            innerphibits_,
                            outerphibits_,
                            innerphimin,
                            innerphimax,
                            outerphimin,
                            outerphimax,
                            innermem,
                            outermem);

  outertememlut.initteptlut(false,
                            true,
                            iSeed_,
                            layerdisk1_,
                            layerdisk2_,
                            innerphibits_,
                            outerphibits_,
                            innerphimin,
                            innerphimax,
                            outerphimin,
                            outerphimax,
                            innermem,
                            outermem);

  innervmstubs_->setbendtable(innertememlut);
  outervmstubs_->setbendtable(outertememlut);
}

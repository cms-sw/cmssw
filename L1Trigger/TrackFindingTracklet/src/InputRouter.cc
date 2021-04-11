#include "L1Trigger/TrackFindingTracklet/interface/InputRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/InputLinkMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/DTCLinkMemory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

InputRouter::InputRouter(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global) {}

void InputRouter::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }

  if (output == "stubout") {
    InputLinkMemory* tmp = dynamic_cast<InputLinkMemory*>(memory);
    assert(tmp != nullptr);
    unsigned int layerdisk = tmp->getName()[4] - '1';
    if (tmp->getName()[3] == 'D') {
      layerdisk += N_LAYER;
    }
    assert(layerdisk < N_LAYER + N_DISK);
    unsigned int phireg = tmp->getName()[8] - 'A';
    std::pair<unsigned int, unsigned int> layerphireg(layerdisk, phireg);
    irstubs_.emplace_back(layerphireg, tmp);
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void InputRouter::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "stubin") {
    dtcstubs_ = dynamic_cast<DTCLinkMemory*>(memory);
    assert(dtcstubs_ != nullptr);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void InputRouter::execute() {
  for (unsigned int i = 0; i < settings_.maxStep("IR"); i++) {
    if (i >= dtcstubs_->nStubs()) {
      break;
    }

    Stub* stub = dtcstubs_->getStub(i);

    unsigned int layerdisk = stub->l1tstub()->layerdisk();

    FPGAWord iphi = stub->phicorr();
    unsigned int iphipos = iphi.value() >> (iphi.nbits() - settings_.nbitsallstubs(layerdisk));

    std::pair<unsigned int, unsigned int> layerphireg(layerdisk, iphipos);

    //Fill inner allstubs memories - in HLS this is the same write to multiple memories
    int iadd = 0;
    for (auto& irstubmem : irstubs_) {
      if (layerphireg == irstubmem.first) {
        irstubmem.second->addStub(stub);
        iadd++;
      }
    }
    assert(iadd == 1);
  }
}

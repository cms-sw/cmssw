#include "L1Trigger/TrackFindingTracklet/interface/ProjectionRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

ProjectionRouter::ProjectionRouter(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global), rinvbendlut_(settings) {
  layerdisk_ = initLayerDisk(3);

  vmprojs_.resize(settings_.nvmme(layerdisk_), nullptr);

  nrbits_ = 5;
  nphiderbits_ = 6;

  if (layerdisk_ >= N_LAYER) {
    rinvbendlut_.initProjectionBend(
        global->ITC_L1L2()->der_phiD_final.K(), layerdisk_ - N_LAYER, nrbits_, nphiderbits_);
  }
}

void ProjectionRouter::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "allprojout") {
    auto* tmp = dynamic_cast<AllProjectionsMemory*>(memory);
    assert(tmp != nullptr);
    allproj_ = tmp;
    return;
  }

  unsigned int nproj = settings_.nallstubs(layerdisk_);
  unsigned int nprojvm = settings_.nvmme(layerdisk_);

  for (unsigned int iproj = 0; iproj < nproj; iproj++) {
    for (unsigned int iprojvm = 0; iprojvm < nprojvm; iprojvm++) {
      std::string name = "vmprojoutPHI";
      name += char(iproj + 'A');
      name += std::to_string(iproj * nprojvm + iprojvm + 1);
      if (output == name) {
        auto* tmp = dynamic_cast<VMProjectionsMemory*>(memory);
        assert(tmp != nullptr);
        vmprojs_[iprojvm] = tmp;
        return;
      }
    }
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find output: " << output;
}

void ProjectionRouter::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input.substr(0, 4) == "proj" && input.substr(input.size() - 2, 2) == "in") {
    auto* tmp = dynamic_cast<TrackletProjectionsMemory*>(memory);
    assert(tmp != nullptr);
    inputproj_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find input: " << input;
}

void ProjectionRouter::execute() {
  unsigned int allprojcount = 0;

  //These are just here to test that the order is correct. Does not affect the actual execution

  int lastTCID = -1;

  for (auto& iproj : inputproj_) {
    for (unsigned int i = 0; i < iproj->nTracklets(); i++) {
      if (allprojcount >= settings_.maxStep("PR"))
        continue;

      Tracklet* tracklet = iproj->getTracklet(i);

      FPGAWord fpgaphi;

      if (layerdisk_ < N_LAYER) {
        fpgaphi = tracklet->proj(layerdisk_).fpgaphiproj();
      } else {
        Projection& proj = tracklet->proj(layerdisk_);
        fpgaphi = proj.fpgaphiproj();

        //The next lines looks up the predicted bend based on:
        // 1 - r projections
        // 2 - phi derivative
        // 3 - the sign - i.e. if track is forward or backward

        int rindex = (proj.fpgarzproj().value() >> (proj.fpgarzproj().nbits() - nrbits_)) & ((1 << nrbits_) - 1);

        int phiderindex = (proj.fpgaphiprojder().value() >> (proj.fpgaphiprojder().nbits() - nphiderbits_)) &
                          ((1 << nphiderbits_) - 1);

        int signindex = (proj.fpgarzprojder().value() < 0);

        int bendindex = (signindex << (nphiderbits_ + nrbits_)) + (rindex << (nphiderbits_)) + phiderindex;

        int ibendproj = rinvbendlut_.lookup(bendindex);

        proj.setBendIndex(ibendproj);
      }

      unsigned int iphivm =
          fpgaphi.bits(fpgaphi.nbits() - settings_.nbitsallstubs(layerdisk_) - settings_.nbitsvmme(layerdisk_),
                       settings_.nbitsvmme(layerdisk_));

      //This block of code just checks that the configuration is consistent
      if (lastTCID >= tracklet->TCID()) {
        edm::LogPrint("Tracklet") << "Wrong TCID ordering for projections in " << getName();
      } else {
        lastTCID = tracklet->TCID();
      }

      allproj_->addTracklet(tracklet);

      vmprojs_[iphivm]->addTracklet(tracklet, allprojcount);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << getName() << " projection to " << vmprojs_[iphivm]->getName() << " iphivm "
                                     << iphivm;
      }

      allprojcount++;
    }
  }

  if (settings_.writeMonitorData("AP")) {
    globals_->ofstream("allprojections.txt") << getName() << " " << allproj_->nTracklets() << endl;
  }

  if (settings_.writeMonitorData("VMP")) {
    ofstream& out = globals_->ofstream("chisq.txt");
    for (unsigned int i = 0; i < 8; i++) {
      if (vmprojs_[i] != nullptr) {
        out << vmprojs_[i]->getName() << " " << vmprojs_[i]->nTracklets() << endl;
      }
    }
  }
}

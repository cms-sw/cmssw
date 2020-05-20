#include "L1Trigger/TrackFindingTracklet/interface/ProjectionRouter.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

ProjectionRouter::ProjectionRouter(string name, const Settings* settings, Globals* global, unsigned int iSector)
    : ProcessBase(name, settings, global, iSector) {
  layerdisk_ = initLayerDisk(3);

  vmprojs_.resize(settings_->nvmme(layerdisk_), nullptr);

  nrbits_ = 5;
  nphiderbits_ = 6;
}

void ProjectionRouter::addOutput(MemoryBase* memory, string output) {
  if (settings_->writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "allprojout") {
    auto* tmp = dynamic_cast<AllProjectionsMemory*>(memory);
    assert(tmp != nullptr);
    allproj_ = tmp;
    return;
  }

  unsigned int nproj = settings_->nallstubs(layerdisk_);
  unsigned int nprojvm = settings_->nvmme(layerdisk_);

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
  if (settings_->writetrace()) {
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
  if (globals_->projectionRouterBendTable() == nullptr) {
    auto* bendTablePtr = new ProjectionRouterBendTable();
    bendTablePtr->init(settings_, globals_, nrbits_, nphiderbits_);
    globals_->projectionRouterBendTable() = bendTablePtr;
  }

  unsigned int allprojcount = 0;

  //These are just here to test that the order is correct. Does not affect the actual execution

  int lastTCID = -1;

  for (auto& iproj : inputproj_) {
    for (unsigned int i = 0; i < iproj->nTracklets(); i++) {
      if (allprojcount > settings_->maxStep("PR"))
        continue;

      Tracklet* tracklet = iproj->getTracklet(i);

      FPGAWord fpgaphi;

      if (layerdisk_ < N_LAYER) {
        fpgaphi = tracklet->fpgaphiproj(layerdisk_ + 1);
      } else {
        int disk = layerdisk_ - (N_LAYER - 1);
        fpgaphi = tracklet->fpgaphiprojdisk(disk);

        //The next lines looks up the predicted bend based on:
        // 1 - r projections
        // 2 - phi derivative
        // 3 - the sign - i.e. if track is forward or backward
        int rindex = (tracklet->fpgarprojdisk(disk).value() >> (tracklet->fpgarprojdisk(disk).nbits() - nrbits_)) &
                     ((1 << nrbits_) - 1);

        int phiderindex = (tracklet->fpgaphiprojderdisk(disk).value() >>
                           (tracklet->fpgaphiprojderdisk(disk).nbits() - nphiderbits_)) &
                          ((1 << nphiderbits_) - 1);

        int signindex = (tracklet->fpgarprojderdisk(disk).value() < 0);

        int bendindex = (signindex << (nphiderbits_ + nrbits_)) + (rindex << (nphiderbits_)) + phiderindex;

        int ibendproj = globals_->projectionRouterBendTable()->bendLoookup(disk - 1, bendindex);

        tracklet->setBendIndex(ibendproj, disk);
      }

      unsigned int iphivm =
          fpgaphi.bits(fpgaphi.nbits() - settings_->nbitsallstubs(layerdisk_) - settings_->nbitsvmme(layerdisk_),
                       settings_->nbitsvmme(layerdisk_));

      //This block of code just checks that the configuration is consistent
      if (lastTCID >= tracklet->TCID()) {
        edm::LogPrint("Tracklet") << "Wrong TCID ordering for projections in " << getName();
      } else {
        lastTCID = tracklet->TCID();
      }

      allproj_->addTracklet(tracklet);

      vmprojs_[iphivm]->addTracklet(tracklet, allprojcount);

      allprojcount++;
    }
  }

  if (settings_->writeMonitorData("AP")) {
    globals_->ofstream("allprojections.txt") << getName() << " " << allproj_->nTracklets() << endl;
  }

  if (settings_->writeMonitorData("VMP")) {
    ofstream& out = globals_->ofstream("chisq.txt");
    for (unsigned int i = 0; i < 8; i++) {
      if (vmprojs_[i] != nullptr) {
        out << vmprojs_[i]->getName() << " " << vmprojs_[i]->nTracklets() << endl;
      }
    }
  }
}

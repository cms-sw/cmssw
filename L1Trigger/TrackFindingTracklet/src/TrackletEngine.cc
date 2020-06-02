#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngine.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace trklet;
using namespace std;

TrackletEngine::TrackletEngine(string name, Settings const& settings, Globals* global, unsigned int iSector)
    : ProcessBase(name, settings, global, iSector) {
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

        if (!(pttableinner_[ptinnerindex] && pttableouter_[ptouterindex])) {
          if (settings_.debugTracklet()) {
            edm::LogVerbatim("Tracklet") << "Stub pair rejected because of stub pt cut bends : "
                                         << benddecode(innervmstub.bend().value(), innervmstub.isPSmodule()) << " "
                                         << benddecode(outervmstub.bend().value(), outervmstub.isPSmodule());
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

  int outerrbits = 3;
  if (iSeed_ < 4) {
    outerrbits = 0;
  }

  int outerrbins = (1 << outerrbits);
  int innerphibins = (1 << innerphibits_);
  int outerphibins = (1 << outerphibits_);

  double innerphimin, innerphimax;
  innervmstubs_->getPhiRange(innerphimin, innerphimax, iSeed_, 0);

  double outerphimin, outerphimax;
  outervmstubs_->getPhiRange(outerphimin, outerphimax, iSeed_, 1);

  double phiinner[2];
  double phiouter[2];
  double router[2];

  unsigned int nbendbitsinner = 3;
  unsigned int nbendbitsouter = 3;
  if (iSeed_ == 2) {
    nbendbitsouter = 4;
  }
  if (iSeed_ == 3) {
    nbendbitsinner = 4;
    nbendbitsouter = 4;
  }

  std::vector<bool> vmbendinner((1 << nbendbitsinner), false);
  std::vector<bool> vmbendouter((1 << nbendbitsouter), false);

  for (int iphiinnerbin = 0; iphiinnerbin < innerphibins; iphiinnerbin++) {
    phiinner[0] = innerphimin + iphiinnerbin * (innerphimax - innerphimin) / innerphibins;
    phiinner[1] = innerphimin + (iphiinnerbin + 1) * (innerphimax - innerphimin) / innerphibins;
    for (int iphiouterbin = 0; iphiouterbin < outerphibins; iphiouterbin++) {
      phiouter[0] = outerphimin + iphiouterbin * (outerphimax - outerphimin) / outerphibins;
      phiouter[1] = outerphimin + (iphiouterbin + 1) * (outerphimax - outerphimin) / outerphibins;
      for (int irouterbin = 0; irouterbin < outerrbins; irouterbin++) {
        if (iSeed_ >= 4) {
          router[0] =
              settings_.rmindiskvm() + irouterbin * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
          router[1] = settings_.rmindiskvm() +
                      (irouterbin + 1) * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
        } else {
          router[0] = settings_.rmean(layerdisk2_);
          router[1] = settings_.rmean(layerdisk2_);
        }

        double bendinnermin = 20.0;
        double bendinnermax = -20.0;
        double bendoutermin = 20.0;
        double bendoutermax = -20.0;
        double rinvmin = 1.0;
        for (int i1 = 0; i1 < 2; i1++) {
          for (int i2 = 0; i2 < 2; i2++) {
            for (int i3 = 0; i3 < 2; i3++) {
              double rinner = 0.0;
              if (iSeed_ == 4 || iSeed_ == 5) {
                rinner = router[i3] * settings_.zmean(layerdisk1_ - N_LAYER) / settings_.zmean(layerdisk2_ - N_LAYER);
              } else {
                rinner = settings_.rmean(layerdisk1_);
              }
              double rinv1 = rinv(phiinner[i1], phiouter[i2], rinner, router[i3]);
              double pitchinner =
                  (rinner < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
              double pitchouter =
                  (router[i3] < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
              double abendinner = -bend(rinner, rinv1, pitchinner);
              double abendouter = -bend(router[i3], rinv1, pitchouter);
              if (abendinner < bendinnermin)
                bendinnermin = abendinner;
              if (abendinner > bendinnermax)
                bendinnermax = abendinner;
              if (abendouter < bendoutermin)
                bendoutermin = abendouter;
              if (abendouter > bendoutermax)
                bendoutermax = abendouter;
              if (std::abs(rinv1) < rinvmin) {
                rinvmin = std::abs(rinv1);
              }
            }
          }
        }

        bool passptcut = rinvmin < settings_.rinvcutte();

        for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
          double bend = benddecode(ibend, nbendbitsinner == 3);

          bool passinner = bend - bendinnermin > -settings_.bendcutte(0, iSeed_) &&
                           bend - bendinnermax < settings_.bendcutte(0, iSeed_);
          if (passinner)
            vmbendinner[ibend] = true;
          pttableinner_.push_back(passinner && passptcut);
        }

        for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
          double bend = benddecode(ibend, nbendbitsouter == 3);

          bool passouter = bend - bendoutermin > -settings_.bendcutte(1, iSeed_) &&
                           bend - bendoutermax < settings_.bendcutte(1, iSeed_);
          if (passouter)
            vmbendouter[ibend] = true;
          pttableouter_.push_back(passouter && passptcut);
        }
      }
    }
  }

  innervmstubs_->setbendtable(vmbendinner);
  outervmstubs_->setbendtable(vmbendouter);

  if (iSector_ == 0 && settings_.writeTable())
    writeTETable();
}

void TrackletEngine::writeTETable() {
  ofstream outstubptinnercut;
  outstubptinnercut.open(getName() + "_stubptinnercut.tab");
  outstubptinnercut << "{" << endl;
  for (unsigned int i = 0; i < pttableinner_.size(); i++) {
    if (i != 0)
      outstubptinnercut << "," << endl;
    outstubptinnercut << pttableinner_[i];
  }
  outstubptinnercut << endl << "};" << endl;
  outstubptinnercut.close();

  ofstream outstubptoutercut;
  outstubptoutercut.open(getName() + "_stubptoutercut.tab");
  outstubptoutercut << "{" << endl;
  for (unsigned int i = 0; i < pttableouter_.size(); i++) {
    if (i != 0)
      outstubptoutercut << "," << endl;
    outstubptoutercut << pttableouter_[i];
  }
  outstubptoutercut << endl << "};" << endl;
  outstubptoutercut.close();
}

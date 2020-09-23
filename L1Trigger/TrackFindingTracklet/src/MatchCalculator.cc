#include "L1Trigger/TrackFindingTracklet/interface/MatchCalculator.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/CandidateMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/FullMatchMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace trklet;

MatchCalculator::MatchCalculator(string name, Settings const& settings, Globals* global, unsigned int iSector)
    : ProcessBase(name, settings, global, iSector) {
  phioffset_ = phimin_;

  phiregion_ = name[8] - 'A';
  layerdisk_ = initLayerDisk(3);

  fullMatches_.resize(12, nullptr);

  //TODO - need to sort out constants here
  icorrshift_ = 7;

  if (layerdisk_ < N_PSLAYER) {
    icorzshift_ = -1 - settings_.PS_zderL_shift();
  } else {
    icorzshift_ = -1 - settings_.SS_zderL_shift();
  }
  phi0shift_ = 3;
  fact_ = 1;
  if (layerdisk_ >= N_PSLAYER && layerdisk_ < N_LAYER) {
    fact_ = (1 << (settings_.nzbitsstub(0) - settings_.nzbitsstub(5)));
    icorrshift_ -= (10 - settings_.nrbitsstub(layerdisk_));
    icorzshift_ += (settings_.nzbitsstub(0) - settings_.nzbitsstub(5) + settings_.nrbitsstub(layerdisk_) -
                    settings_.nrbitsstub(0));
    phi0shift_ = 0;
  }

  for (unsigned int iSeed = 0; iSeed < N_SEED; iSeed++) {
    if (layerdisk_ < N_LAYER) {
      phimatchcut_[iSeed] =
          settings_.rphimatchcut(iSeed, layerdisk_) / (settings_.kphi1() * settings_.rmean(layerdisk_));
      zmatchcut_[iSeed] = settings_.zmatchcut(iSeed, layerdisk_) / settings_.kz();
    } else {
      rphicutPS_[iSeed] = settings_.rphicutPS(iSeed, layerdisk_ - N_LAYER) / (settings_.kphi() * settings_.kr());
      rphicut2S_[iSeed] = settings_.rphicut2S(iSeed, layerdisk_ - N_LAYER) / (settings_.kphi() * settings_.kr());
      rcut2S_[iSeed] = settings_.rcut2S(iSeed, layerdisk_ - N_LAYER) / settings_.krprojshiftdisk();
      rcutPS_[iSeed] = settings_.rcutPS(iSeed, layerdisk_ - N_LAYER) / settings_.krprojshiftdisk();
    }
  }

  if (iSector_ == 0 && layerdisk_ < N_LAYER && settings_.writeTable()) {
    ofstream outphicut;
    outphicut.open(settings_.tablePath()+getName() + "_phicut.tab");
    outphicut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outphicut << "," << endl;
      outphicut << phimatchcut_[seedindex];
    }
    outphicut << endl << "};" << endl;
    outphicut.close();

    ofstream outzcut;
    outzcut.open(settings_.tablePath()+getName() + "_zcut.tab");
    outzcut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outzcut << "," << endl;
      outzcut << zmatchcut_[seedindex];
    }
    outzcut << endl << "};" << endl;
    outzcut.close();
  }

  if (iSector_ == 0 && layerdisk_ >= N_LAYER && settings_.writeTable()) {
    ofstream outphicut;
    outphicut.open(settings_.tablePath()+getName() + "_PSphicut.tab");
    outphicut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outphicut << "," << endl;
      outphicut << rphicutPS_[seedindex];
    }
    outphicut << endl << "};" << endl;
    outphicut.close();

    outphicut.open(settings_.tablePath()+getName() + "_2Sphicut.tab");
    outphicut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outphicut << "," << endl;
      outphicut << rphicut2S_[seedindex];
    }
    outphicut << endl << "};" << endl;
    outphicut.close();

    ofstream outzcut;
    outzcut.open(settings_.tablePath()+getName() + "_PSrcut.tab");
    outzcut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outzcut << "," << endl;
      outzcut << rcutPS_[seedindex];
    }
    outzcut << endl << "};" << endl;
    outzcut.close();

    outzcut.open(settings_.tablePath()+getName() + "_2Srcut.tab");
    outzcut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outzcut << "," << endl;
      outzcut << rcut2S_[seedindex];
    }
    outzcut << endl << "};" << endl;
    outzcut.close();
  }

  for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
    ialphafactinner_[i] = (1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                          (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSinner(i) * settings_.rDSSinner(i)) /
                          settings_.kphi();
    ialphafactouter_[i] = (1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                          (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSouter(i) * settings_.rDSSouter(i)) /
                          settings_.kphi();
  }
}

void MatchCalculator::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output.substr(0, 8) == "matchout") {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    unsigned int iSeed = getISeed(memory->getName());
    fullMatches_[iSeed] = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find output " << output;
}

void MatchCalculator::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "allstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    allstubs_ = tmp;
    return;
  }
  if (input == "allprojin") {
    auto* tmp = dynamic_cast<AllProjectionsMemory*>(memory);
    assert(tmp != nullptr);
    allprojs_ = tmp;
    return;
  }
  if (input.substr(0, 5) == "match" && input.substr(input.size() - 2, 2) == "in") {
    auto* tmp = dynamic_cast<CandidateMatchMemory*>(memory);
    assert(tmp != nullptr);
    matches_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find input " << input;
}

void MatchCalculator::execute() {
  unsigned int countall = 0;
  unsigned int countsel = 0;

  Tracklet* oldTracklet = nullptr;

  std::vector<std::pair<std::pair<Tracklet*, int>, const Stub*> > mergedMatches = mergeMatches(matches_);

  for (unsigned int j = 0; j < mergedMatches.size(); j++) {
    if (settings_.debugTracklet() && j == 0) {
      edm::LogVerbatim("Tracklet") << getName() << " has " << mergedMatches.size() << " candidate matches";
    }

    countall++;

    const Stub* fpgastub = mergedMatches[j].second;
    Tracklet* tracklet = mergedMatches[j].first.first;
    const L1TStub* stub = fpgastub->l1tstub();

    //check that the matches are orderd correctly
    //allow equal here since we can have more than one cadidate match per tracklet projection
    if (oldTracklet != nullptr) {
      assert(oldTracklet->TCID() <= tracklet->TCID());
    }
    oldTracklet = tracklet;

    if (layerdisk_ < N_LAYER) {
      //Integer calculation

      int ir = fpgastub->r().value();
      int iphi = tracklet->fpgaphiproj(layerdisk_ + 1).value();
      int icorr = (ir * tracklet->fpgaphiprojder(layerdisk_ + 1).value()) >> icorrshift_;
      iphi += icorr;

      int iz = tracklet->fpgazproj(layerdisk_ + 1).value();
      int izcor = (ir * tracklet->fpgazprojder(layerdisk_ + 1).value() + (1 << (icorzshift_ - 1))) >> icorzshift_;
      iz += izcor;

      int ideltaz = fpgastub->z().value() - iz;
      int ideltaphi = (fpgastub->phi().value() << phi0shift_) - (iphi << (settings_.phi0bitshift() - 1 + phi0shift_));

      //Floating point calculations

      double phi = stub->phi() - phioffset_;
      double r = stub->r();
      double z = stub->z();

      if (settings_.useapprox()) {
        double dphi = reco::reduceRange(phi - fpgastub->phiapprox(0.0, 0.0));
        assert(std::abs(dphi) < 0.001);
        phi = fpgastub->phiapprox(0.0, 0.0);
        z = fpgastub->zapprox();
        r = fpgastub->rapprox();
      }

      if (phi < 0)
        phi += 2 * M_PI;

      double dr = r - tracklet->rproj(layerdisk_ + 1);
      assert(std::abs(dr) < settings_.drmax());

      double dphi =
          reco::reduceRange(phi - (tracklet->phiproj(layerdisk_ + 1) + dr * tracklet->phiprojder(layerdisk_ + 1)));

      double dz = z - (tracklet->zproj(layerdisk_ + 1) + dr * tracklet->zprojder(layerdisk_ + 1));

      double dphiapprox = reco::reduceRange(
          phi - (tracklet->phiprojapprox(layerdisk_ + 1) + dr * tracklet->phiprojderapprox(layerdisk_ + 1)));

      double dzapprox = z - (tracklet->zprojapprox(layerdisk_ + 1) + dr * tracklet->zprojderapprox(layerdisk_ + 1));

      int seedindex = tracklet->getISeed();

      assert(phimatchcut_[seedindex] > 0);
      assert(zmatchcut_[seedindex] > 0);

      if (settings_.bookHistos()) {
        bool truthmatch = tracklet->stubtruthmatch(stub);

        HistBase* hists = globals_->histograms();
        hists->FillLayerResidual(layerdisk_ + 1,
                                 seedindex,
                                 dphiapprox * settings_.rmean(layerdisk_),
                                 ideltaphi * settings_.kphi1() * settings_.rmean(layerdisk_),
                                 ideltaz * fact_ * settings_.kz(),
                                 dz,
                                 truthmatch);
      }

      if (std::abs(dphi) > 0.2 || std::abs(dphiapprox) > 0.2) {
        edm::LogProblem("Tracklet") << "WARNING dphi and/or dphiapprox too large : " << dphi << " " << dphiapprox
                                    << endl;
      }
      assert(std::abs(dphi) < 0.2);
      assert(std::abs(dphiapprox) < 0.2);

      if (settings_.writeMonitorData("Residuals")) {
        double pt = 0.01 * settings_.c() * settings_.bfield() / std::abs(tracklet->rinv());

        globals_->ofstream("layerresiduals.txt")
            << layerdisk_ + 1 << " " << seedindex << " " << pt << " "
            << ideltaphi * settings_.kphi1() * settings_.rmean(layerdisk_) << " "
            << dphiapprox * settings_.rmean(layerdisk_) << " "
            << phimatchcut_[seedindex] * settings_.kphi1() * settings_.rmean(layerdisk_) << "   "
            << ideltaz * fact_ * settings_.kz() << " " << dz << " " << zmatchcut_[seedindex] * settings_.kz() << endl;
      }

      bool imatch = (std::abs(ideltaphi) <= (int)phimatchcut_[seedindex]) &&
                    (std::abs(ideltaz * fact_) <= (int)zmatchcut_[seedindex]);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << getName() << " imatch = " << imatch << " ideltaphi cut " << ideltaphi << " "
                                     << phimatchcut_[seedindex] << " ideltaz*fact cut " << ideltaz * fact_ << " "
                                     << zmatchcut_[seedindex];
      }

      if (imatch) {
        countsel++;

        tracklet->addMatch(layerdisk_ + 1,
                           ideltaphi,
                           ideltaz,
                           dphi,
                           dz,
                           dphiapprox,
                           dzapprox,
                           (phiregion_ << 7) + fpgastub->stubindex().value(),
                           stub->r(),
                           mergedMatches[j].second);

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "Accepted full match in layer " << getName() << " " << tracklet << " "
                                       << iSector_;
        }

        fullMatches_[seedindex]->addMatch(tracklet, mergedMatches[j].second);
      }
    } else {  //disk matches

      //check that stubs and projections in same half of detector
      assert(stub->z() * tracklet->t() > 0.0);

      int sign = (tracklet->t() > 0.0) ? 1 : -1;
      int disk = sign * (layerdisk_ - (N_LAYER - 1));
      assert(disk != 0);

      //Perform integer calculations here

      int iz = fpgastub->z().value();
      int iphi = tracklet->fpgaphiprojdisk(disk).value();

      //TODO - need to express interms of constants
      int shifttmp = 6;
      int iphicorr = (iz * tracklet->fpgaphiprojderdisk(disk).value()) >> shifttmp;

      iphi += iphicorr;

      int ir = tracklet->fpgarprojdisk(disk).value();

      //TODO - need to express interms of constants
      int shifttmp2 = 7;
      int ircorr = (iz * tracklet->fpgarprojderdisk(disk).value()) >> shifttmp2;

      ir += ircorr;

      int ideltaphi = fpgastub->phi().value() * settings_.kphi() / settings_.kphi() - iphi;

      int irstub = fpgastub->r().value();
      int ialphafact = 0;
      if (!stub->isPSmodule()) {
        assert(irstub < (int)N_DSS_MOD * 2);
        if (abs(disk) <= 2) {
          ialphafact = ialphafactinner_[irstub];
          irstub = settings_.rDSSinner(irstub) / settings_.kr();
        } else {
          ialphafact = ialphafactouter_[irstub];
          irstub = settings_.rDSSouter(irstub) / settings_.kr();
        }
      }

      //TODO stub and projection r should not use different # bits...
      int ideltar = (irstub >> 1) - ir;

      if (!stub->isPSmodule()) {
        int ialphanew = fpgastub->alphanew().value();
        int iphialphacor = ((ideltar * ialphanew * ialphafact) >> settings_.alphashift());
        ideltaphi += iphialphacor;
      }

      //Perform floating point calculations here

      double phi = stub->phi() - phioffset_;
      double z = stub->z();
      double r = stub->r();

      if (settings_.useapprox()) {
        double dphi = reco::reduceRange(phi - fpgastub->phiapprox(0.0, 0.0));
        assert(std::abs(dphi) < 0.001);
        phi = fpgastub->phiapprox(0.0, 0.0);
        z = fpgastub->zapprox();
        r = fpgastub->rapprox();
      }

      if (phi < 0)
        phi += 2 * M_PI;

      double dz = z - sign * settings_.zmean(layerdisk_ - N_LAYER);

      if (std::abs(dz) > settings_.dzmax()) {
        throw cms::Exception("LogicError")
            << __FILE__ << " " << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed()
            << "\n stub " << stub->z() << " disk " << disk << " " << dz;
      }

      double phiproj = tracklet->phiprojdisk(disk) + dz * tracklet->phiprojderdisk(disk);

      double rproj = tracklet->rprojdisk(disk) + dz * tracklet->rprojderdisk(disk);

      double deltar = r - rproj;

      double dr = stub->r() - rproj;

      double dphi = reco::reduceRange(phi - phiproj);

      double dphiapprox =
          reco::reduceRange(phi - (tracklet->phiprojapproxdisk(disk) + dz * tracklet->phiprojderapproxdisk(disk)));

      double drapprox = stub->r() - (tracklet->rprojapproxdisk(disk) + dz * tracklet->rprojderapproxdisk(disk));

      double drphi = dphi * stub->r();
      double drphiapprox = dphiapprox * stub->r();

      if (!stub->isPSmodule()) {
        double alphanorm = stub->alphanorm();
        dphi += dr * alphanorm * settings_.half2SmoduleWidth() / stub->r2();
        dphiapprox += drapprox * alphanorm * settings_.half2SmoduleWidth() / stub->r2();

        drphi += dr * alphanorm * settings_.half2SmoduleWidth() / stub->r();
        drphiapprox += dr * alphanorm * settings_.half2SmoduleWidth() / stub->r();
      }

      int seedindex = tracklet->getISeed();

      int idrphicut = rphicutPS_[seedindex];
      int idrcut = rcutPS_[seedindex];
      if (!stub->isPSmodule()) {
        idrphicut = rphicut2S_[seedindex];
        idrcut = rcut2S_[seedindex];
      }

      double drphicut = idrphicut * settings_.kphi() * settings_.kr();
      double drcut = idrcut * settings_.krprojshiftdisk();

      if (settings_.writeMonitorData("Residuals")) {
        double pt = 0.01 * settings_.c() * settings_.bfield() / std::abs(tracklet->rinv());

        globals_->ofstream("diskresiduals.txt")
            << disk << " " << stub->isPSmodule() << " " << tracklet->layer() << " " << abs(tracklet->disk()) << " "
            << pt << " " << ideltaphi * settings_.kphi() * stub->r() << " " << drphiapprox << " " << drphicut << " "
            << ideltar * settings_.krprojshiftdisk() << " " << deltar << " " << drcut << " " << endl;
      }

      bool match = (std::abs(drphi) < drphicut) && (std::abs(deltar) < drcut);

      bool imatch = (std::abs(ideltaphi * irstub) < idrphicut) && (std::abs(ideltar) < idrcut);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "imatch match disk: " << imatch << " " << match << " " << std::abs(ideltaphi)
                                     << " " << drphicut / (settings_.kphi() * stub->r()) << " " << std::abs(ideltar)
                                     << " " << drcut / settings_.krprojshiftdisk() << " r = " << stub->r();
      }

      if (imatch) {
        countsel++;

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "MatchCalculator found match in disk " << getName();
        }

        if (std::abs(dphi) >= 0.25) {
          edm::LogVerbatim("Tracklet") << "dphi " << dphi << " Seed / ISeed " << tracklet->getISeed();
        }
        assert(std::abs(dphi) < 0.25);
        assert(std::abs(dphiapprox) < 0.25);

        tracklet->addMatchDisk(disk,
                               ideltaphi,
                               ideltar,
                               drphi / stub->r(),
                               dr,
                               drphiapprox / stub->r(),
                               drapprox,
                               stub->alpha(settings_.stripPitch(stub->isPSmodule())),
                               (phiregion_ << 7) + fpgastub->stubindex().value(),
                               stub->z(),
                               fpgastub);
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "Accepted full match in disk " << getName() << " " << tracklet << " "
                                       << iSector_;
        }

        fullMatches_[seedindex]->addMatch(tracklet, mergedMatches[j].second);
      }
    }
    if (countall >= settings_.maxStep("MC"))
      break;
  }

  if (settings_.writeMonitorData("MC")) {
    globals_->ofstream("matchcalculator.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}

std::vector<std::pair<std::pair<Tracklet*, int>, const Stub*> > MatchCalculator::mergeMatches(
    vector<CandidateMatchMemory*>& candmatch) {
  std::vector<std::pair<std::pair<Tracklet*, int>, const Stub*> > tmp;

  std::vector<unsigned int> indexArray;
  indexArray.reserve(candmatch.size());
  for (unsigned int i = 0; i < candmatch.size(); i++) {
    indexArray.push_back(0);
  }

  int bestIndex = -1;
  do {
    int bestSector = 100;
    int bestTCID = (1 << 16);
    bestIndex = -1;
    for (unsigned int i = 0; i < candmatch.size(); i++) {
      if (indexArray[i] >= candmatch[i]->nMatches()) {
        // skip as we were at the end
        continue;
      }
      int TCID = candmatch[i]->getMatch(indexArray[i]).first.first->TCID();
      int dSector = 0;
      if (dSector > 2)
        dSector -= N_SECTOR;
      if (dSector < -2)
        dSector += N_SECTOR;
      assert(abs(dSector) < 2);
      if (dSector == -1)
        dSector = 2;
      if (dSector < bestSector) {
        bestSector = dSector;
        bestTCID = TCID;
        bestIndex = i;
      }
      if (dSector == bestSector) {
        if (TCID < bestTCID) {
          bestTCID = TCID;
          bestIndex = i;
        }
      }
    }
    if (bestIndex != -1) {
      tmp.push_back(candmatch[bestIndex]->getMatch(indexArray[bestIndex]));
      indexArray[bestIndex]++;
    }
  } while (bestIndex != -1);

  if (layerdisk_ < N_LAYER) {
    int lastTCID = -1;
    bool error = false;

    //Allow equal TCIDs since we can have multiple candidate matches
    for (unsigned int i = 1; i < tmp.size(); i++) {
      if (lastTCID > tmp[i].first.first->TCID()) {
        edm::LogProblem("Tracklet") << "Wrong TCID ordering for projections in " << getName() << " last " << lastTCID
                                    << " " << tmp[i].first.first->TCID();
        error = true;
      } else {
        lastTCID = tmp[i].first.first->TCID();
      }
    }

    if (error) {
      for (unsigned int i = 1; i < tmp.size(); i++) {
        edm::LogProblem("Tracklet") << "Wrong order for in " << getName() << " " << i << " " << tmp[i].first.first
                                    << " " << tmp[i].first.first->TCID();
      }
    }
  }

  for (unsigned int i = 0; i < tmp.size(); i++) {
    if (i > 0) {
      //This allows for equal TCIDs. This means that we can e.g. have a track seeded
      //in L1L2 that projects to both L3 and D4. The algorithm will pick up the first hit and
      //drop the second

      assert(tmp[i - 1].first.first->TCID() <= tmp[i].first.first->TCID());
    }
  }

  return tmp;
}

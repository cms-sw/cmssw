#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace trklet;

TrackletCalculatorDisplaced::TrackletCalculatorDisplaced(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global) {
  for (unsigned int ilayer = 0; ilayer < N_LAYER; ilayer++) {
    vector<TrackletProjectionsMemory*> tmp(settings.nallstubs(ilayer), nullptr);
    trackletprojlayers_.push_back(tmp);
  }

  for (unsigned int idisk = 0; idisk < N_DISK; idisk++) {
    vector<TrackletProjectionsMemory*> tmp(settings.nallstubs(idisk + N_LAYER), nullptr);
    trackletprojdisks_.push_back(tmp);
  }

  layer_ = 0;
  disk_ = 0;

  string name1 = name.substr(1);  //this is to correct for "TCD" having one more letter then "TC"
  if (name1[3] == 'L')
    layer_ = name1[4] - '0';
  if (name1[3] == 'D')
    disk_ = name1[4] - '0';

  // set TC index
  iSeed_ = 0;

  int iTC = name1[9] - 'A';

  if (name1.substr(3, 6) == "L3L4L2")
    iSeed_ = 8;
  else if (name1.substr(3, 6) == "L5L6L4")
    iSeed_ = 9;
  else if (name1.substr(3, 6) == "L2L3D1")
    iSeed_ = 10;
  else if (name1.substr(3, 6) == "D1D2L2")
    iSeed_ = 11;

  assert(iSeed_ != 0);

  TCIndex_ = (iSeed_ << 4) + iTC;
  assert(TCIndex_ >= 128 && TCIndex_ < 191);

  assert((layer_ != 0) || (disk_ != 0));

  toR_.clear();
  toZ_.clear();

  if (iSeed_ == 8 || iSeed_ == 9) {
    if (layer_ == 3) {
      rzmeanInv_[0] = 1.0 / settings_.rmean(2 - 1);
      rzmeanInv_[1] = 1.0 / settings_.rmean(3 - 1);
      rzmeanInv_[2] = 1.0 / settings_.rmean(4 - 1);

      rproj_[0] = settings_.rmean(0);
      rproj_[1] = settings_.rmean(4);
      rproj_[2] = settings_.rmean(5);
      lproj_[0] = 1;
      lproj_[1] = 5;
      lproj_[2] = 6;

      dproj_[0] = 1;
      dproj_[1] = 2;
      dproj_[2] = 0;
      toZ_.push_back(settings_.zmean(0));
      toZ_.push_back(settings_.zmean(1));
    }
    if (layer_ == 5) {
      rzmeanInv_[0] = 1.0 / settings_.rmean(4 - 1);
      rzmeanInv_[1] = 1.0 / settings_.rmean(5 - 1);
      rzmeanInv_[2] = 1.0 / settings_.rmean(6 - 1);

      rproj_[0] = settings_.rmean(0);
      rproj_[1] = settings_.rmean(1);
      rproj_[2] = settings_.rmean(2);
      lproj_[0] = 1;
      lproj_[1] = 2;
      lproj_[2] = 3;

      dproj_[0] = 0;
      dproj_[1] = 0;
      dproj_[2] = 0;
    }
    for (unsigned int i = 0; i < N_LAYER - 3; ++i)
      toR_.push_back(rproj_[i]);
  }

  if (iSeed_ == 10 || iSeed_ == 11) {
    if (layer_ == 2) {
      rzmeanInv_[0] = 1.0 / settings_.rmean(2 - 1);
      rzmeanInv_[1] = 1.0 / settings_.rmean(3 - 1);
      rzmeanInv_[2] = 1.0 / settings_.zmean(1 - 1);

      rproj_[0] = settings_.rmean(0);
      lproj_[0] = 1;
      lproj_[1] = -1;
      lproj_[2] = -1;

      zproj_[0] = settings_.zmean(1);
      zproj_[1] = settings_.zmean(2);
      zproj_[2] = settings_.zmean(3);
      dproj_[0] = 2;
      dproj_[1] = 3;
      dproj_[2] = 4;
    }
    if (disk_ == 1) {
      rzmeanInv_[0] = 1.0 / settings_.rmean(2 - 1);
      rzmeanInv_[1] = 1.0 / settings_.zmean(1 - 1);
      rzmeanInv_[2] = 1.0 / settings_.zmean(2 - 1);

      rproj_[0] = settings_.rmean(0);
      lproj_[0] = 1;
      lproj_[1] = -1;
      lproj_[2] = -1;

      zproj_[0] = settings_.zmean(2);
      zproj_[1] = settings_.zmean(3);
      zproj_[2] = settings_.zmean(4);
      dproj_[0] = 3;
      dproj_[1] = 4;
      dproj_[2] = 5;
    }
    toR_.push_back(settings_.rmean(0));
    for (unsigned int i = 0; i < N_DISK - 2; ++i)
      toZ_.push_back(zproj_[i]);
  }
}

void TrackletCalculatorDisplaced::addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory) {
  outputProj = dynamic_cast<TrackletProjectionsMemory*>(memory);
  assert(outputProj != nullptr);
}

void TrackletCalculatorDisplaced::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }

  if (output == "trackpar") {
    auto* tmp = dynamic_cast<TrackletParametersMemory*>(memory);
    assert(tmp != nullptr);
    trackletpars_ = tmp;
    return;
  }

  if (output.substr(0, 7) == "projout") {
    //output is on the form 'projoutL2PHIC' or 'projoutD3PHIB'
    auto* tmp = dynamic_cast<TrackletProjectionsMemory*>(memory);
    assert(tmp != nullptr);

    unsigned int layerdisk = output[8] - '1';   //layer or disk counting from 0
    unsigned int phiregion = output[12] - 'A';  //phiregion counting from 0

    if (output[7] == 'L') {
      assert(layerdisk < N_LAYER);
      assert(phiregion < trackletprojlayers_[layerdisk].size());
      //check that phiregion not already initialized
      assert(trackletprojlayers_[layerdisk][phiregion] == nullptr);
      trackletprojlayers_[layerdisk][phiregion] = tmp;
      return;
    }

    if (output[7] == 'D') {
      assert(layerdisk < N_DISK);
      assert(phiregion < trackletprojdisks_[layerdisk].size());
      //check that phiregion not already initialized
      assert(trackletprojdisks_[layerdisk][phiregion] == nullptr);
      trackletprojdisks_[layerdisk][phiregion] = tmp;
      return;
    }
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void TrackletCalculatorDisplaced::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }

  if (input == "thirdallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    innerallstubs_.push_back(tmp);
    return;
  }
  if (input == "firstallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    middleallstubs_.push_back(tmp);
    return;
  }
  if (input == "secondallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    outerallstubs_.push_back(tmp);
    return;
  }
  if (input.find("stubtriplet") == 0) {
    auto* tmp = dynamic_cast<StubTripletsMemory*>(memory);
    assert(tmp != nullptr);
    stubtriplets_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TrackletCalculatorDisplaced::execute(unsigned int iSector, double phimin, double phimax) {
  unsigned int countall = 0;
  unsigned int countsel = 0;

  phimin_ = phimin;
  phimax_ = phimax;
  iSector_ = iSector;

  for (auto& stubtriplet : stubtriplets_) {
    if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
      edm::LogVerbatim("Tracklet") << "Will break on too many tracklets in " << getName();
      break;
    }
    for (unsigned int i = 0; i < stubtriplet->nStubTriplets(); i++) {
      countall++;

      const Stub* innerFPGAStub = stubtriplet->getFPGAStub1(i);
      const L1TStub* innerStub = innerFPGAStub->l1tstub();

      const Stub* middleFPGAStub = stubtriplet->getFPGAStub2(i);
      const L1TStub* middleStub = middleFPGAStub->l1tstub();

      const Stub* outerFPGAStub = stubtriplet->getFPGAStub3(i);
      const L1TStub* outerStub = outerFPGAStub->l1tstub();

      if (settings_.debugTracklet())
        edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced execute " << getName() << "[" << iSector_ << "]";

      if (innerFPGAStub->layerdisk() < N_LAYER && middleFPGAStub->layerdisk() < N_LAYER &&
          outerFPGAStub->layerdisk() < N_LAYER) {
        //barrel+barrel seeding
        bool accept = LLLSeeding(innerFPGAStub, innerStub, middleFPGAStub, middleStub, outerFPGAStub, outerStub);
        if (accept)
          countsel++;
      } else if (innerFPGAStub->layerdisk() >= N_LAYER && middleFPGAStub->layerdisk() >= N_LAYER &&
                 outerFPGAStub->layerdisk() >= N_LAYER) {
        throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";
      } else {
        //layer+disk seeding
        if (innerFPGAStub->layerdisk() < N_LAYER && middleFPGAStub->layerdisk() >= N_LAYER &&
            outerFPGAStub->layerdisk() >= N_LAYER) {  //D1D2L2
          bool accept = DDLSeeding(innerFPGAStub, innerStub, middleFPGAStub, middleStub, outerFPGAStub, outerStub);
          if (accept)
            countsel++;
        } else if (innerFPGAStub->layerdisk() >= N_LAYER && middleFPGAStub->layerdisk() < N_LAYER &&
                   outerFPGAStub->layerdisk() < N_LAYER) {  //L2L3D1
          bool accept = LLDSeeding(innerFPGAStub, innerStub, middleFPGAStub, middleStub, outerFPGAStub, outerStub);
          if (accept)
            countsel++;
        } else {
          throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";
        }
      }

      if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
        edm::LogVerbatim("Tracklet") << "Will break on number of tracklets in " << getName();
        break;
      }

      if (countall >= settings_.maxStep("TC")) {
        if (settings_.debugTracklet())
          edm::LogVerbatim("Tracklet") << "Will break on MAXTC 1";
        break;
      }
      if (settings_.debugTracklet())
        edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced execute done";
    }
    if (countall >= settings_.maxStep("TC")) {
      if (settings_.debugTracklet())
        edm::LogVerbatim("Tracklet") << "Will break on MAXTC 2";
      break;
    }
  }

  if (settings_.writeMonitorData("TPD")) {
    globals_->ofstream("trackletcalculatordisplaced.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}

void TrackletCalculatorDisplaced::addDiskProj(Tracklet* tracklet, int disk) {
  disk = std::abs(disk);
  FPGAWord fpgar = tracklet->proj(N_LAYER + disk - 1).fpgarzproj();

  if (fpgar.value() * settings_.krprojshiftdisk() < settings_.rmindiskvm())
    return;
  if (fpgar.value() * settings_.krprojshiftdisk() > settings_.rmaxdisk())
    return;

  FPGAWord fpgaphi = tracklet->proj(N_LAYER + disk - 1).fpgaphiproj();

  int iphivmRaw = fpgaphi.value() >> (fpgaphi.nbits() - 5);
  int iphi = iphivmRaw / (32 / settings_.nallstubs(disk + N_LAYER - 1));

  addProjectionDisk(disk, iphi, trackletprojdisks_[disk - 1][iphi], tracklet);
}

bool TrackletCalculatorDisplaced::addLayerProj(Tracklet* tracklet, int layer) {
  assert(layer > 0);

  FPGAWord fpgaz = tracklet->proj(layer - 1).fpgarzproj();
  FPGAWord fpgaphi = tracklet->proj(layer - 1).fpgaphiproj();

  if (fpgaz.atExtreme())
    return false;

  if (std::abs(fpgaz.value() * settings_.kz()) > settings_.zlength())
    return false;

  int iphivmRaw = fpgaphi.value() >> (fpgaphi.nbits() - 5);
  int iphi = iphivmRaw / (32 / settings_.nallstubs(layer - 1));

  addProjection(layer, iphi, trackletprojlayers_[layer - 1][iphi], tracklet);

  return true;
}

void TrackletCalculatorDisplaced::addProjection(int layer,
                                                int iphi,
                                                TrackletProjectionsMemory* trackletprojs,
                                                Tracklet* tracklet) {
  if (trackletprojs == nullptr) {
    if (settings_.warnNoMem()) {
      edm::LogVerbatim("Tracklet") << "No projection memory exists in " << getName() << " for layer = " << layer
                                   << " iphi = " << iphi + 1;
    }
    return;
  }
  assert(trackletprojs != nullptr);
  trackletprojs->addProj(tracklet);
}

void TrackletCalculatorDisplaced::addProjectionDisk(int disk,
                                                    int iphi,
                                                    TrackletProjectionsMemory* trackletprojs,
                                                    Tracklet* tracklet) {
  if (trackletprojs == nullptr) {
    if (layer_ == 3 && abs(disk) == 3)
      return;  //L3L4 projections to D3 are not used.
    if (settings_.warnNoMem()) {
      edm::LogVerbatim("Tracklet") << "No projection memory exists in " << getName() << " for disk = " << abs(disk)
                                   << " iphi = " << iphi + 1;
    }
    return;
  }
  assert(trackletprojs != nullptr);
  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << getName() << " adding projection to " << trackletprojs->getName();
  trackletprojs->addProj(tracklet);
}

bool TrackletCalculatorDisplaced::LLLSeeding(const Stub* innerFPGAStub,
                                             const L1TStub* innerStub,
                                             const Stub* middleFPGAStub,
                                             const L1TStub* middleStub,
                                             const Stub* outerFPGAStub,
                                             const L1TStub* outerStub) {
  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced " << getName() << " " << layer_
                                 << " trying stub triplet in layer (L L L): " << innerFPGAStub->layer().value() << " "
                                 << middleFPGAStub->layer().value() << " " << outerFPGAStub->layer().value();

  assert(outerFPGAStub->layerdisk() < N_LAYER);

  double r1 = innerStub->r();
  double z1 = innerStub->z();
  double phi1 = innerStub->phi();

  double r2 = middleStub->r();
  double z2 = middleStub->z();
  double phi2 = middleStub->phi();

  double r3 = outerStub->r();
  double z3 = outerStub->z();
  double phi3 = outerStub->phi();

  int take3 = 0;
  if (layer_ == 5)
    take3 = 1;
  unsigned ndisks = 0;

  double rinv, phi0, d0, t, z0;

  Projection projs[N_LAYER + N_DISK];

  double phiproj[N_LAYER - 2], zproj[N_LAYER - 2], phider[N_LAYER - 2], zder[N_LAYER - 2];
  double phiprojdisk[N_DISK], rprojdisk[N_DISK], phiderdisk[N_DISK], rderdisk[N_DISK];

  exacttracklet(r1,
                z1,
                phi1,
                r2,
                z2,
                phi2,
                r3,
                z3,
                phi3,
                take3,
                rinv,
                phi0,
                d0,
                t,
                z0,
                phiproj,
                zproj,
                phiprojdisk,
                rprojdisk,
                phider,
                zder,
                phiderdisk,
                rderdisk);
  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << __LINE__ << ":" << __FILE__ << "LLL Exact values " << innerFPGAStub->isBarrel()
                                 << middleFPGAStub->isBarrel() << outerFPGAStub->isBarrel() << " " << phi1 << ", " << z1
                                 << ", " << r1 << ", " << phi2 << ", " << z2 << ", " << r2 << ", " << phi3 << ", " << z3
                                 << ", " << r3 << endl;

  if (settings_.useapprox()) {
    phi1 = innerFPGAStub->phiapprox(phimin_, phimax_);
    z1 = innerFPGAStub->zapprox();
    r1 = innerFPGAStub->rapprox();

    phi2 = middleFPGAStub->phiapprox(phimin_, phimax_);
    z2 = middleFPGAStub->zapprox();
    r2 = middleFPGAStub->rapprox();

    phi3 = outerFPGAStub->phiapprox(phimin_, phimax_);
    z3 = outerFPGAStub->zapprox();
    r3 = outerFPGAStub->rapprox();
  }

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << __LINE__ << ":" << __FILE__ << "LLL Approx values " << innerFPGAStub->isBarrel()
                                 << middleFPGAStub->isBarrel() << outerFPGAStub->isBarrel() << " " << phi1 << ", " << z1
                                 << ", " << r1 << ", " << phi2 << ", " << z2 << ", " << r2 << ", " << phi3 << ", " << z3
                                 << ", " << r3 << endl;

  double rinvapprox, phi0approx, d0approx, tapprox, z0approx;
  double phiprojapprox[N_LAYER - 2], zprojapprox[N_LAYER - 2], phiderapprox[N_LAYER - 2], zderapprox[N_LAYER - 2];
  double phiprojdiskapprox[N_DISK], rprojdiskapprox[N_DISK];
  double phiderdiskapprox[N_DISK], rderdiskapprox[N_DISK];

  //TODO: implement the actual integer calculation
  if (settings_.useapprox()) {
    approxtracklet(r1,
                   z1,
                   phi1,
                   r2,
                   z2,
                   phi2,
                   r3,
                   z3,
                   phi3,
                   take3,
                   ndisks,
                   rinvapprox,
                   phi0approx,
                   d0approx,
                   tapprox,
                   z0approx,
                   phiprojapprox,
                   zprojapprox,
                   phiderapprox,
                   zderapprox,
                   phiprojdiskapprox,
                   rprojdiskapprox,
                   phiderdiskapprox,
                   rderdiskapprox);
  } else {
    rinvapprox = rinv;
    phi0approx = phi0;
    d0approx = d0;
    tapprox = t;
    z0approx = z0;

    for (unsigned int i = 0; i < toR_.size(); ++i) {
      phiprojapprox[i] = phiproj[i];
      zprojapprox[i] = zproj[i];
      phiderapprox[i] = phider[i];
      zderapprox[i] = zder[i];
    }

    for (unsigned int i = 0; i < toZ_.size(); ++i) {
      phiprojdiskapprox[i] = phiprojdisk[i];
      rprojdiskapprox[i] = rprojdisk[i];
      phiderdiskapprox[i] = phiderdisk[i];
      rderdiskapprox[i] = rderdisk[i];
    }
  }

  //store the approcximate results

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "rinvapprox: " << rinvapprox << " rinv: " << rinv << endl;
    edm::LogVerbatim("Tracklet") << "phi0approx: " << phi0approx << " phi0: " << phi0 << endl;
    edm::LogVerbatim("Tracklet") << "d0approx: " << d0approx << " d0: " << d0 << endl;
    edm::LogVerbatim("Tracklet") << "tapprox: " << tapprox << " t: " << t << endl;
    edm::LogVerbatim("Tracklet") << "z0approx: " << z0approx << " z0: " << z0 << endl;
  }

  for (unsigned int i = 0; i < toR_.size(); ++i) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "phiprojapprox[" << i << "]: " << phiprojapprox[i] << " phiproj[" << i
                                   << "]: " << phiproj[i] << endl;
      edm::LogVerbatim("Tracklet") << "zprojapprox[" << i << "]: " << zprojapprox[i] << " zproj[" << i
                                   << "]: " << zproj[i] << endl;
      edm::LogVerbatim("Tracklet") << "phiderapprox[" << i << "]: " << phiderapprox[i] << " phider[" << i
                                   << "]: " << phider[i] << endl;
      edm::LogVerbatim("Tracklet") << "zderapprox[" << i << "]: " << zderapprox[i] << " zder[" << i << "]: " << zder[i]
                                   << endl;
    }
  }

  for (unsigned int i = 0; i < toZ_.size(); ++i) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "phiprojdiskapprox[" << i << "]: " << phiprojdiskapprox[i] << " phiprojdisk[" << i
                                   << "]: " << phiprojdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "rprojdiskapprox[" << i << "]: " << rprojdiskapprox[i] << " rprojdisk[" << i
                                   << "]: " << rprojdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "phiderdiskapprox[" << i << "]: " << phiderdiskapprox[i] << " phiderdisk[" << i
                                   << "]: " << phiderdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "rderdiskapprox[" << i << "]: " << rderdiskapprox[i] << " rderdisk[" << i
                                   << "]: " << rderdisk[i] << endl;
    }
  }

  //now binary
  double krinv = settings_.kphi1() / settings_.kr() * pow(2, settings_.rinv_shift()),
         kphi0 = settings_.kphi1() * pow(2, settings_.phi0_shift()),
         kt = settings_.kz() / settings_.kr() * pow(2, settings_.t_shift()),
         kz0 = settings_.kz() * pow(2, settings_.z0_shift()),
         kphiproj = settings_.kphi1() * pow(2, settings_.SS_phiL_shift()),
         kphider = settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderL_shift()),
         kzproj = settings_.kz() * pow(2, settings_.PS_zL_shift()),
         kzder = settings_.kz() / settings_.kr() * pow(2, settings_.PS_zderL_shift()),
         kphiprojdisk = settings_.kphi1() * pow(2, settings_.SS_phiD_shift()),
         kphiderdisk = settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderD_shift()),
         krprojdisk = settings_.kr() * pow(2, settings_.PS_rD_shift()),
         krderdisk = settings_.kr() / settings_.kz() * pow(2, settings_.PS_rderD_shift());

  int irinv, iphi0, id0, it, iz0;
  int iphiproj[N_LAYER - 2], izproj[N_LAYER - 2], iphider[N_LAYER - 2], izder[N_LAYER - 2];
  int iphiprojdisk[N_DISK], irprojdisk[N_DISK], iphiderdisk[N_DISK], irderdisk[N_DISK];

  //store the binary results
  irinv = rinvapprox / krinv;
  iphi0 = phi0approx / kphi0;
  id0 = d0approx / settings_.kd0();
  it = tapprox / kt;
  iz0 = z0approx / kz0;

  bool success = true;
  if (std::abs(rinvapprox) > settings_.rinvcut()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "TrackletCalculator::LLL Seeding irinv too large: " << rinvapprox << "(" << irinv
                                   << ")";
    success = false;
  }
  if (std::abs(z0approx) > settings_.disp_z0cut()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet z0 cut " << z0approx << " in layer " << layer_;
    success = false;
  }
  if (std::abs(d0approx) > settings_.maxd0()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet approx d0 cut " << d0approx;
    success = false;
  }

  if (std::abs(d0) > settings_.maxd0()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet exact d0 cut " << d0;
    success = false;
  }

  if (!success) {
    return false;
  }

  double phicritapprox = phi0approx - asin((0.5 * settings_.rcrit() * rinvapprox) + (d0approx / settings_.rcrit()));
  int phicrit = iphi0 - 2 * irinv - 2 * id0;

  int iphicritmincut = settings_.phicritminmc() / globals_->ITC_L1L2()->phi0_final.K();
  int iphicritmaxcut = settings_.phicritmaxmc() / globals_->ITC_L1L2()->phi0_final.K();

  bool keepapprox = (phicritapprox > settings_.phicritminmc()) && (phicritapprox < settings_.phicritmaxmc()),
       keep = (phicrit > iphicritmincut) && (phicrit < iphicritmaxcut);

  if (settings_.debugTracklet())
    if (keep && !keepapprox)
      edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced::LLLSeeding tracklet kept with exact phicrit cut "
                                      "but not approximate, phicritapprox: "
                                   << phicritapprox;
  if (settings_.usephicritapprox()) {
    if (!keepapprox) {
      return false;
    }
  } else {
    if (!keep) {
      return false;
    }
  }

  for (unsigned int i = 0; i < toR_.size(); ++i) {
    iphiproj[i] = phiprojapprox[i] / kphiproj;
    izproj[i] = zprojapprox[i] / kzproj;

    iphider[i] = phiderapprox[i] / kphider;
    izder[i] = zderapprox[i] / kzder;

    //check that z projection is in range
    if (izproj[i] < -(1 << (settings_.nzbitsstub(0) - 1)))
      continue;
    if (izproj[i] >= (1 << (settings_.nzbitsstub(0) - 1)))
      continue;

    //check that phi projection is in range
    if (iphiproj[i] >= (1 << settings_.nphibitsstub(N_LAYER - 1)) - 1)
      continue;
    if (iphiproj[i] <= 0)
      continue;

    //adjust number of bits for phi and z projection
    if (rproj_[i] < settings_.rPS2S()) {
      iphiproj[i] >>= (settings_.nphibitsstub(N_LAYER - 1) - settings_.nphibitsstub(0));
      if (iphiproj[i] >= (1 << settings_.nphibitsstub(0)) - 1)
        iphiproj[i] = (1 << settings_.nphibitsstub(0)) - 2;  //-2 not to hit atExtreme
    } else {
      izproj[i] >>= (settings_.nzbitsstub(0) - settings_.nzbitsstub(N_LAYER - 1));
    }

    if (rproj_[i] < settings_.rPS2S()) {
      if (iphider[i] < -(1 << (settings_.nbitsphiprojderL123() - 1))) {
        iphider[i] = -(1 << (settings_.nbitsphiprojderL123() - 1));
      }
      if (iphider[i] >= (1 << (settings_.nbitsphiprojderL123() - 1))) {
        iphider[i] = (1 << (settings_.nbitsphiprojderL123() - 1)) - 1;
      }
    } else {
      if (iphider[i] < -(1 << (settings_.nbitsphiprojderL456() - 1))) {
        iphider[i] = -(1 << (settings_.nbitsphiprojderL456() - 1));
      }
      if (iphider[i] >= (1 << (settings_.nbitsphiprojderL456() - 1))) {
        iphider[i] = (1 << (settings_.nbitsphiprojderL456() - 1)) - 1;
      }
    }

    projs[lproj_[i] - 1].init(settings_,
                              lproj_[i] - 1,
                              iphiproj[i],
                              izproj[i],
                              iphider[i],
                              izder[i],
                              phiproj[i],
                              zproj[i],
                              phider[i],
                              zder[i],
                              phiprojapprox[i],
                              zprojapprox[i],
                              phiderapprox[i],
                              zderapprox[i],
                              false);
  }

  if (std::abs(it * kt) > 1.0) {
    for (unsigned int i = 0; i < toZ_.size(); ++i) {
      iphiprojdisk[i] = phiprojdiskapprox[i] / kphiprojdisk;
      irprojdisk[i] = rprojdiskapprox[i] / krprojdisk;

      iphiderdisk[i] = phiderdiskapprox[i] / kphiderdisk;
      irderdisk[i] = rderdiskapprox[i] / krderdisk;

      //check phi projection in range
      if (iphiprojdisk[i] <= 0)
        continue;
      if (iphiprojdisk[i] >= (1 << settings_.nphibitsstub(0)) - 1)
        continue;

      //check r projection in range
      if (rprojdiskapprox[i] < settings_.rmindisk() || rprojdiskapprox[i] >= settings_.rmaxdisk())
        continue;

      projs[N_LAYER + i].init(settings_,
                              N_LAYER + i,
                              iphiprojdisk[i],
                              irprojdisk[i],
                              iphiderdisk[i],
                              irderdisk[i],
                              phiprojdisk[i],
                              rprojdisk[i],
                              phiderdisk[i],
                              rderdisk[i],
                              phiprojdiskapprox[i],
                              rprojdiskapprox[i],
                              phiderdisk[i],
                              rderdisk[i],
                              false);
    }
  }

  if (settings_.writeMonitorData("TrackletPars")) {
    globals_->ofstream("trackletpars.txt")
        << layer_ << " , " << rinv << " , " << rinvapprox << " , " << phi0 << " , " << phi0approx << " , " << t << " , "
        << tapprox << " , " << z0 << " , " << z0approx << " , " << d0 << " , " << d0approx << endl;
  }

  Tracklet* tracklet = new Tracklet(settings_,
                                    iSeed_,
                                    innerFPGAStub,
                                    middleFPGAStub,
                                    outerFPGAStub,
                                    rinv,
                                    phi0,
                                    d0,
                                    z0,
                                    t,
                                    rinvapprox,
                                    phi0approx,
                                    d0approx,
                                    z0approx,
                                    tapprox,
                                    irinv,
                                    iphi0,
                                    id0,
                                    iz0,
                                    it,
                                    projs,
                                    false);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced " << getName()
                                 << " Found LLL tracklet in sector = " << iSector_ << " phi0 = " << phi0;

  tracklet->setTrackletIndex(trackletpars_->nTracklets());
  tracklet->setTCIndex(TCIndex_);

  if (settings_.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::app);
    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
    fout.close();
  }
  trackletpars_->addTracklet(tracklet);

  bool addL5 = false;
  bool addL6 = false;
  for (unsigned int j = 0; j < toR_.size(); j++) {
    bool added = false;

    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "adding layer projection " << j << "/" << toR_.size() << " " << lproj_[j];
    if (tracklet->validProj(lproj_[j] - 1)) {
      added = addLayerProj(tracklet, lproj_[j]);
      if (added && lproj_[j] == 5)
        addL5 = true;
      if (added && lproj_[j] == 6)
        addL6 = true;
    }
  }

  for (unsigned int j = 0; j < toZ_.size(); j++) {
    int disk = dproj_[j];
    if (disk == 0)
      continue;
    if (disk == 2 && addL5)
      continue;
    if (disk == 1 && addL6)
      continue;
    if (it < 0)
      disk = -disk;
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "adding disk projection " << j << "/" << toZ_.size() << " " << disk;
    if (tracklet->validProj(N_LAYER + abs(disk) - 1)) {
      addDiskProj(tracklet, disk);
    }
  }

  return true;
}

bool TrackletCalculatorDisplaced::DDLSeeding(const Stub* innerFPGAStub,
                                             const L1TStub* innerStub,
                                             const Stub* middleFPGAStub,
                                             const L1TStub* middleStub,
                                             const Stub* outerFPGAStub,
                                             const L1TStub* outerStub) {
  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced " << getName() << " " << layer_
                                 << " trying stub triplet in  (L2 D1 D2): " << innerFPGAStub->layer().value() << " "
                                 << middleFPGAStub->disk().value() << " " << outerFPGAStub->disk().value();

  int take3 = 1;  //D1D2L2
  unsigned ndisks = 2;

  double r1 = innerStub->r();
  double z1 = innerStub->z();
  double phi1 = innerStub->phi();

  double r2 = middleStub->r();
  double z2 = middleStub->z();
  double phi2 = middleStub->phi();

  double r3 = outerStub->r();
  double z3 = outerStub->z();
  double phi3 = outerStub->phi();

  double rinv, phi0, d0, t, z0;

  double phiproj[N_LAYER - 2], zproj[N_LAYER - 2], phider[N_LAYER - 2], zder[N_LAYER - 2];
  double phiprojdisk[N_DISK], rprojdisk[N_DISK], phiderdisk[N_DISK], rderdisk[N_DISK];

  exacttracklet(r1,
                z1,
                phi1,
                r2,
                z2,
                phi2,
                r3,
                z3,
                phi3,
                take3,
                rinv,
                phi0,
                d0,
                t,
                z0,
                phiproj,
                zproj,
                phiprojdisk,
                rprojdisk,
                phider,
                zder,
                phiderdisk,
                rderdisk);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << __LINE__ << ":" << __FILE__ << " DLL Exact values " << innerFPGAStub->isBarrel()
                                 << middleFPGAStub->isBarrel() << outerFPGAStub->isBarrel() << " " << phi1 << ", " << z1
                                 << ", " << r1 << ", " << phi2 << ", " << z2 << ", " << r2 << ", " << phi3 << ", " << z3
                                 << ", " << r3 << endl;

  if (settings_.useapprox()) {
    phi1 = innerFPGAStub->phiapprox(phimin_, phimax_);
    z1 = innerFPGAStub->zapprox();
    r1 = innerFPGAStub->rapprox();

    phi2 = middleFPGAStub->phiapprox(phimin_, phimax_);
    z2 = middleFPGAStub->zapprox();
    r2 = middleFPGAStub->rapprox();

    phi3 = outerFPGAStub->phiapprox(phimin_, phimax_);
    z3 = outerFPGAStub->zapprox();
    r3 = outerFPGAStub->rapprox();
  }

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << __LINE__ << ":" << __FILE__ << "DLL Approx values " << innerFPGAStub->isBarrel()
                                 << middleFPGAStub->isBarrel() << outerFPGAStub->isBarrel() << " " << phi1 << ", " << z1
                                 << ", " << r1 << ", " << phi2 << ", " << z2 << ", " << r2 << ", " << phi3 << ", " << z3
                                 << ", " << r3 << endl;

  double rinvapprox, phi0approx, d0approx, tapprox, z0approx;
  double phiprojapprox[N_LAYER - 2], zprojapprox[N_LAYER - 2], phiderapprox[N_LAYER - 2], zderapprox[N_LAYER - 2];
  double phiprojdiskapprox[N_DISK], rprojdiskapprox[N_DISK];
  double phiderdiskapprox[N_DISK], rderdiskapprox[N_DISK];

  //TODO: implement the actual integer calculation
  if (settings_.useapprox()) {
    approxtracklet(r1,
                   z1,
                   phi1,
                   r2,
                   z2,
                   phi2,
                   r3,
                   z3,
                   phi3,
                   take3,
                   ndisks,
                   rinvapprox,
                   phi0approx,
                   d0approx,
                   tapprox,
                   z0approx,
                   phiprojapprox,
                   zprojapprox,
                   phiderapprox,
                   zderapprox,
                   phiprojdiskapprox,
                   rprojdiskapprox,
                   phiderdiskapprox,
                   rderdiskapprox);
  } else {
    rinvapprox = rinv;
    phi0approx = phi0;
    d0approx = d0;
    tapprox = t;
    z0approx = z0;

    for (unsigned int i = 0; i < toR_.size(); ++i) {
      phiprojapprox[i] = phiproj[i];
      zprojapprox[i] = zproj[i];
      phiderapprox[i] = phider[i];
      zderapprox[i] = zder[i];
    }

    for (unsigned int i = 0; i < toZ_.size(); ++i) {
      phiprojdiskapprox[i] = phiprojdisk[i];
      rprojdiskapprox[i] = rprojdisk[i];
      phiderdiskapprox[i] = phiderdisk[i];
      rderdiskapprox[i] = rderdisk[i];
    }
  }

  //store the approcximate results
  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "rinvapprox: " << rinvapprox << " rinv: " << rinv << endl;
    edm::LogVerbatim("Tracklet") << "phi0approx: " << phi0approx << " phi0: " << phi0 << endl;
    edm::LogVerbatim("Tracklet") << "d0approx: " << d0approx << " d0: " << d0 << endl;
    edm::LogVerbatim("Tracklet") << "tapprox: " << tapprox << " t: " << t << endl;
    edm::LogVerbatim("Tracklet") << "z0approx: " << z0approx << " z0: " << z0 << endl;
  }

  for (unsigned int i = 0; i < toR_.size(); ++i) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "phiprojapprox[" << i << "]: " << phiprojapprox[i] << " phiproj[" << i
                                   << "]: " << phiproj[i] << endl;
      edm::LogVerbatim("Tracklet") << "zprojapprox[" << i << "]: " << zprojapprox[i] << " zproj[" << i
                                   << "]: " << zproj[i] << endl;
      edm::LogVerbatim("Tracklet") << "phiderapprox[" << i << "]: " << phiderapprox[i] << " phider[" << i
                                   << "]: " << phider[i] << endl;
      edm::LogVerbatim("Tracklet") << "zderapprox[" << i << "]: " << zderapprox[i] << " zder[" << i << "]: " << zder[i]
                                   << endl;
    }
  }

  for (unsigned int i = 0; i < toZ_.size(); ++i) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "phiprojdiskapprox[" << i << "]: " << phiprojdiskapprox[i] << " phiprojdisk[" << i
                                   << "]: " << phiprojdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "rprojdiskapprox[" << i << "]: " << rprojdiskapprox[i] << " rprojdisk[" << i
                                   << "]: " << rprojdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "phiderdiskapprox[" << i << "]: " << phiderdiskapprox[i] << " phiderdisk[" << i
                                   << "]: " << phiderdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "rderdiskapprox[" << i << "]: " << rderdiskapprox[i] << " rderdisk[" << i
                                   << "]: " << rderdisk[i] << endl;
    }
  }

  //now binary
  double krinv = settings_.kphi1() / settings_.kr() * pow(2, settings_.rinv_shift()),
         kphi0 = settings_.kphi1() * pow(2, settings_.phi0_shift()),
         kt = settings_.kz() / settings_.kr() * pow(2, settings_.t_shift()),
         kz0 = settings_.kz() * pow(2, settings_.z0_shift()),
         kphiproj = settings_.kphi1() * pow(2, settings_.SS_phiL_shift()),
         kphider = settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderL_shift()),
         kzproj = settings_.kz() * pow(2, settings_.PS_zL_shift()),
         kzder = settings_.kz() / settings_.kr() * pow(2, settings_.PS_zderL_shift()),
         kphiprojdisk = settings_.kphi1() * pow(2, settings_.SS_phiD_shift()),
         kphiderdisk = settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderD_shift()),
         krprojdisk = settings_.kr() * pow(2, settings_.PS_rD_shift()),
         krderdisk = settings_.kr() / settings_.kz() * pow(2, settings_.PS_rderD_shift());

  int irinv, iphi0, id0, it, iz0;
  int iphiproj[N_LAYER - 2], izproj[N_LAYER - 2], iphider[N_LAYER - 2], izder[N_LAYER - 2];
  int iphiprojdisk[N_DISK], irprojdisk[N_DISK], iphiderdisk[N_DISK], irderdisk[N_DISK];

  //store the binary results
  irinv = rinvapprox / krinv;
  iphi0 = phi0approx / kphi0;
  id0 = d0approx / settings_.kd0();
  it = tapprox / kt;
  iz0 = z0approx / kz0;

  bool success = true;
  if (std::abs(rinvapprox) > settings_.rinvcut()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "TrackletCalculator::DDL Seeding irinv too large: " << rinvapprox << "(" << irinv
                                   << ")";
    success = false;
  }
  if (std::abs(z0approx) > settings_.disp_z0cut()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet z0 cut " << z0approx;
    success = false;
  }
  if (std::abs(d0approx) > settings_.maxd0()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet d0 cut " << d0approx;
    success = false;
  }

  if (!success)
    return false;

  double phicritapprox = phi0approx - asin((0.5 * settings_.rcrit() * rinvapprox) + (d0approx / settings_.rcrit()));
  int phicrit = iphi0 - 2 * irinv - 2 * id0;

  int iphicritmincut = settings_.phicritminmc() / globals_->ITC_L1L2()->phi0_final.K();
  int iphicritmaxcut = settings_.phicritmaxmc() / globals_->ITC_L1L2()->phi0_final.K();

  bool keepapprox = (phicritapprox > settings_.phicritminmc()) && (phicritapprox < settings_.phicritmaxmc()),
       keep = (phicrit > iphicritmincut) && (phicrit < iphicritmaxcut);

  if (settings_.debugTracklet())
    if (keep && !keepapprox)
      edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced::DDLSeeding tracklet kept with exact phicrit cut "
                                      "but not approximate, phicritapprox: "
                                   << phicritapprox;
  if (settings_.usephicritapprox()) {
    if (!keepapprox)
      return false;
  } else {
    if (!keep)
      return false;
  }

  Projection projs[N_LAYER + N_DISK];

  for (unsigned int i = 0; i < toR_.size(); ++i) {
    iphiproj[i] = phiprojapprox[i] / kphiproj;
    izproj[i] = zprojapprox[i] / kzproj;

    iphider[i] = phiderapprox[i] / kphider;
    izder[i] = zderapprox[i] / kzder;

    //check that z projection in range
    if (izproj[i] < -(1 << (settings_.nzbitsstub(0) - 1)))
      continue;
    if (izproj[i] >= (1 << (settings_.nzbitsstub(0) - 1)))
      continue;

    //check that phi projection in range
    if (iphiproj[i] >= (1 << settings_.nphibitsstub(5)) - 1)
      continue;
    if (iphiproj[i] <= 0)
      continue;

    if (rproj_[i] < settings_.rPS2S()) {
      iphiproj[i] >>= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));
    } else {
      izproj[i] >>= (settings_.nzbitsstub(0) - settings_.nzbitsstub(5));
    }

    if (rproj_[i] < settings_.rPS2S()) {
      if (iphider[i] < -(1 << (settings_.nbitsphiprojderL123() - 1)))
        iphider[i] = -(1 << (settings_.nbitsphiprojderL123() - 1));
      if (iphider[i] >= (1 << (settings_.nbitsphiprojderL123() - 1)))
        iphider[i] = (1 << (settings_.nbitsphiprojderL123() - 1)) - 1;
    } else {
      if (iphider[i] < -(1 << (settings_.nbitsphiprojderL456() - 1)))
        iphider[i] = -(1 << (settings_.nbitsphiprojderL456() - 1));
      if (iphider[i] >= (1 << (settings_.nbitsphiprojderL456() - 1)))
        iphider[i] = (1 << (settings_.nbitsphiprojderL456() - 1)) - 1;
    }

    projs[lproj_[i] - 1].init(settings_,
                              lproj_[i] - 1,
                              iphiproj[i],
                              izproj[i],
                              iphider[i],
                              izder[i],
                              phiproj[i],
                              zproj[i],
                              phider[i],
                              zder[i],
                              phiprojapprox[i],
                              zprojapprox[i],
                              phiderapprox[i],
                              zderapprox[i],
                              false);
  }

  if (std::abs(it * kt) > 1.0) {
    for (unsigned int i = 0; i < toZ_.size(); ++i) {
      iphiprojdisk[i] = phiprojdiskapprox[i] / kphiprojdisk;
      irprojdisk[i] = rprojdiskapprox[i] / krprojdisk;

      iphiderdisk[i] = phiderdiskapprox[i] / kphiderdisk;
      irderdisk[i] = rderdiskapprox[i] / krderdisk;

      if (iphiprojdisk[i] <= 0)
        continue;
      if (iphiprojdisk[i] >= (1 << settings_.nphibitsstub(0)) - 1)
        continue;

      if (irprojdisk[i] < settings_.rmindisk() / krprojdisk || irprojdisk[i] >= settings_.rmaxdisk() / krprojdisk)
        continue;

      projs[N_LAYER + i + 2].init(settings_,
                                  N_LAYER + i + 2,
                                  iphiprojdisk[i],
                                  irprojdisk[i],
                                  iphiderdisk[i],
                                  irderdisk[i],
                                  phiprojdisk[i],
                                  rprojdisk[i],
                                  phiderdisk[i],
                                  rderdisk[i],
                                  phiprojdiskapprox[i],
                                  rprojdiskapprox[i],
                                  phiderdisk[i],
                                  rderdisk[i],
                                  false);
    }
  }

  if (settings_.writeMonitorData("TrackletPars")) {
    globals_->ofstream("trackletpars.txt")
        << layer_ << " , " << rinv << " , " << rinvapprox << " , " << phi0 << " , " << phi0approx << " , " << t << " , "
        << tapprox << " , " << z0 << " , " << z0approx << " , " << d0 << " , " << d0approx << endl;
  }

  Tracklet* tracklet = new Tracklet(settings_,
                                    iSeed_,
                                    innerFPGAStub,
                                    middleFPGAStub,
                                    outerFPGAStub,
                                    rinv,
                                    phi0,
                                    d0,
                                    z0,
                                    t,
                                    rinvapprox,
                                    phi0approx,
                                    d0approx,
                                    z0approx,
                                    tapprox,
                                    irinv,
                                    iphi0,
                                    id0,
                                    iz0,
                                    it,
                                    projs,
                                    true);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced " << getName()
                                 << " Found DDL tracklet in sector = " << iSector_ << " phi0 = " << phi0;

  tracklet->setTrackletIndex(trackletpars_->nTracklets());
  tracklet->setTCIndex(TCIndex_);

  if (settings_.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::app);
    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
    fout.close();
  }
  trackletpars_->addTracklet(tracklet);

  for (unsigned int j = 0; j < toR_.size(); j++) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "adding layer projection " << j << "/" << toR_.size() << " " << lproj_[j] << " "
                                   << tracklet->validProj(lproj_[j] - 1);
    if (tracklet->validProj(lproj_[j] - 1)) {
      addLayerProj(tracklet, lproj_[j]);
    }
  }

  for (unsigned int j = 0; j < toZ_.size(); j++) {
    int disk = dproj_[j];
    if (disk == 0)
      continue;
    if (it < 0)
      disk = -disk;
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "adding disk projection " << j << "/" << toZ_.size() << " " << disk << " "
                                   << tracklet->validProj(N_LAYER + abs(disk) - 1);
    if (tracklet->validProj(N_LAYER + abs(disk) - 1)) {
      addDiskProj(tracklet, disk);
    }
  }

  return true;
}

bool TrackletCalculatorDisplaced::LLDSeeding(const Stub* innerFPGAStub,
                                             const L1TStub* innerStub,
                                             const Stub* middleFPGAStub,
                                             const L1TStub* middleStub,
                                             const Stub* outerFPGAStub,
                                             const L1TStub* outerStub) {
  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced " << getName() << " " << layer_
                                 << " trying stub triplet in  (L2L3D1): " << middleFPGAStub->layer().value() << " "
                                 << outerFPGAStub->layer().value() << " " << innerFPGAStub->disk().value();

  int take3 = 0;  //L2L3D1
  unsigned ndisks = 1;

  double r3 = innerStub->r();
  double z3 = innerStub->z();
  double phi3 = innerStub->phi();

  double r1 = middleStub->r();
  double z1 = middleStub->z();
  double phi1 = middleStub->phi();

  double r2 = outerStub->r();
  double z2 = outerStub->z();
  double phi2 = outerStub->phi();

  double rinv, phi0, d0, t, z0;

  double phiproj[N_LAYER - 2], zproj[N_LAYER - 2], phider[N_LAYER - 2], zder[N_LAYER - 2];
  double phiprojdisk[N_DISK], rprojdisk[N_DISK], phiderdisk[N_DISK], rderdisk[N_DISK];

  exacttracklet(r1,
                z1,
                phi1,
                r2,
                z2,
                phi2,
                r3,
                z3,
                phi3,
                take3,
                rinv,
                phi0,
                d0,
                t,
                z0,
                phiproj,
                zproj,
                phiprojdisk,
                rprojdisk,
                phider,
                zder,
                phiderdisk,
                rderdisk);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << __LINE__ << ":" << __FILE__ << "LLD Exact values " << innerFPGAStub->isBarrel()
                                 << middleFPGAStub->isBarrel() << outerFPGAStub->isBarrel() << " " << phi3 << ", " << z3
                                 << ", " << r3 << ", " << phi1 << ", " << z1 << ", " << r1 << ", " << phi2 << ", " << z2
                                 << ", " << r2 << endl;

  if (settings_.useapprox()) {
    phi3 = innerFPGAStub->phiapprox(phimin_, phimax_);
    z3 = innerFPGAStub->zapprox();
    r3 = innerFPGAStub->rapprox();

    phi1 = middleFPGAStub->phiapprox(phimin_, phimax_);
    z1 = middleFPGAStub->zapprox();
    r1 = middleFPGAStub->rapprox();

    phi2 = outerFPGAStub->phiapprox(phimin_, phimax_);
    z2 = outerFPGAStub->zapprox();
    r2 = outerFPGAStub->rapprox();
  }

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << __LINE__ << ":" << __FILE__ << "LLD approx values " << innerFPGAStub->isBarrel()
                                 << middleFPGAStub->isBarrel() << outerFPGAStub->isBarrel() << " " << phi3 << ", " << z3
                                 << ", " << r3 << ", " << phi1 << ", " << z1 << ", " << r1 << ", " << phi2 << ", " << z2
                                 << ", " << r2 << endl;

  double rinvapprox, phi0approx, d0approx, tapprox, z0approx;
  double phiprojapprox[N_LAYER - 2], zprojapprox[N_LAYER - 2], phiderapprox[N_LAYER - 2], zderapprox[N_LAYER - 2];
  double phiprojdiskapprox[N_DISK], rprojdiskapprox[N_DISK];
  double phiderdiskapprox[N_DISK], rderdiskapprox[N_DISK];

  //TODO: implement the actual integer calculation
  if (settings_.useapprox()) {
    approxtracklet(r1,
                   z1,
                   phi1,
                   r2,
                   z2,
                   phi2,
                   r3,
                   z3,
                   phi3,
                   take3,
                   ndisks,
                   rinvapprox,
                   phi0approx,
                   d0approx,
                   tapprox,
                   z0approx,
                   phiprojapprox,
                   zprojapprox,
                   phiderapprox,
                   zderapprox,
                   phiprojdiskapprox,
                   rprojdiskapprox,
                   phiderdiskapprox,
                   rderdiskapprox);
  } else {
    rinvapprox = rinv;
    phi0approx = phi0;
    d0approx = d0;
    tapprox = t;
    z0approx = z0;

    for (unsigned int i = 0; i < toR_.size(); ++i) {
      phiprojapprox[i] = phiproj[i];
      zprojapprox[i] = zproj[i];
      phiderapprox[i] = phider[i];
      zderapprox[i] = zder[i];
    }

    for (unsigned int i = 0; i < toZ_.size(); ++i) {
      phiprojdiskapprox[i] = phiprojdisk[i];
      rprojdiskapprox[i] = rprojdisk[i];
      phiderdiskapprox[i] = phiderdisk[i];
      rderdiskapprox[i] = rderdisk[i];
    }
  }

  //store the approcximate results
  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "rinvapprox: " << rinvapprox << " rinv: " << rinv << endl;
    edm::LogVerbatim("Tracklet") << "phi0approx: " << phi0approx << " phi0: " << phi0 << endl;
    edm::LogVerbatim("Tracklet") << "d0approx: " << d0approx << " d0: " << d0 << endl;
    edm::LogVerbatim("Tracklet") << "tapprox: " << tapprox << " t: " << t << endl;
    edm::LogVerbatim("Tracklet") << "z0approx: " << z0approx << " z0: " << z0 << endl;
  }

  for (unsigned int i = 0; i < toR_.size(); ++i) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "phiprojapprox[" << i << "]: " << phiprojapprox[i] << " phiproj[" << i
                                   << "]: " << phiproj[i] << endl;
      edm::LogVerbatim("Tracklet") << "zprojapprox[" << i << "]: " << zprojapprox[i] << " zproj[" << i
                                   << "]: " << zproj[i] << endl;
      edm::LogVerbatim("Tracklet") << "phiderapprox[" << i << "]: " << phiderapprox[i] << " phider[" << i
                                   << "]: " << phider[i] << endl;
      edm::LogVerbatim("Tracklet") << "zderapprox[" << i << "]: " << zderapprox[i] << " zder[" << i << "]: " << zder[i]
                                   << endl;
    }
  }

  for (unsigned int i = 0; i < toZ_.size(); ++i) {
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "phiprojdiskapprox[" << i << "]: " << phiprojdiskapprox[i] << " phiprojdisk[" << i
                                   << "]: " << phiprojdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "rprojdiskapprox[" << i << "]: " << rprojdiskapprox[i] << " rprojdisk[" << i
                                   << "]: " << rprojdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "phiderdiskapprox[" << i << "]: " << phiderdiskapprox[i] << " phiderdisk[" << i
                                   << "]: " << phiderdisk[i] << endl;
      edm::LogVerbatim("Tracklet") << "rderdiskapprox[" << i << "]: " << rderdiskapprox[i] << " rderdisk[" << i
                                   << "]: " << rderdisk[i] << endl;
    }
  }

  //now binary
  double krinv = settings_.kphi1() / settings_.kr() * pow(2, settings_.rinv_shift()),
         kphi0 = settings_.kphi1() * pow(2, settings_.phi0_shift()),
         kt = settings_.kz() / settings_.kr() * pow(2, settings_.t_shift()),
         kz0 = settings_.kz() * pow(2, settings_.z0_shift()),
         kphiproj = settings_.kphi1() * pow(2, settings_.SS_phiL_shift()),
         kphider = settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderL_shift()),
         kzproj = settings_.kz() * pow(2, settings_.PS_zL_shift()),
         kzder = settings_.kz() / settings_.kr() * pow(2, settings_.PS_zderL_shift()),
         kphiprojdisk = settings_.kphi1() * pow(2, settings_.SS_phiD_shift()),
         kphiderdisk = settings_.kphi1() / settings_.kr() * pow(2, settings_.SS_phiderD_shift()),
         krprojdisk = settings_.kr() * pow(2, settings_.PS_rD_shift()),
         krderdisk = settings_.kr() / settings_.kz() * pow(2, settings_.PS_rderD_shift());

  int irinv, iphi0, id0, it, iz0;
  int iphiproj[N_LAYER - 2], izproj[N_LAYER - 2], iphider[N_LAYER - 2], izder[N_LAYER - 2];
  int iphiprojdisk[N_DISK], irprojdisk[N_DISK], iphiderdisk[N_DISK], irderdisk[N_DISK];

  //store the binary results
  irinv = rinvapprox / krinv;
  iphi0 = phi0approx / kphi0;
  id0 = d0approx / settings_.kd0();
  it = tapprox / kt;
  iz0 = z0approx / kz0;

  bool success = true;
  if (std::abs(rinvapprox) > settings_.rinvcut()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "TrackletCalculator:: LLD Seeding irinv too large: " << rinvapprox << "(" << irinv
                                   << ")";
    success = false;
  }
  if (std::abs(z0approx) > settings_.disp_z0cut()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet z0 cut " << z0approx;
    success = false;
  }
  if (std::abs(d0approx) > settings_.maxd0()) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "Failed tracklet d0 cut " << d0approx;
    success = false;
  }

  if (!success)
    return false;

  double phicritapprox = phi0approx - asin((0.5 * settings_.rcrit() * rinvapprox) + (d0approx / settings_.rcrit()));
  int phicrit = iphi0 - 2 * irinv - 2 * id0;

  int iphicritmincut = settings_.phicritminmc() / globals_->ITC_L1L2()->phi0_final.K();
  int iphicritmaxcut = settings_.phicritmaxmc() / globals_->ITC_L1L2()->phi0_final.K();

  bool keepapprox = (phicritapprox > settings_.phicritminmc()) && (phicritapprox < settings_.phicritmaxmc()),
       keep = (phicrit > iphicritmincut) && (phicrit < iphicritmaxcut);

  if (settings_.debugTracklet())
    if (keep && !keepapprox)
      edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced::LLDSeeding tracklet kept with exact phicrit cut "
                                      "but not approximate, phicritapprox: "
                                   << phicritapprox;
  if (settings_.usephicritapprox()) {
    if (!keepapprox)
      return false;
  } else {
    if (!keep)
      return false;
  }

  Projection projs[N_LAYER + N_DISK];

  for (unsigned int i = 0; i < toR_.size(); ++i) {
    iphiproj[i] = phiprojapprox[i] / kphiproj;
    izproj[i] = zprojapprox[i] / kzproj;

    iphider[i] = phiderapprox[i] / kphider;
    izder[i] = zderapprox[i] / kzder;

    if (izproj[i] < -(1 << (settings_.nzbitsstub(0) - 1)))
      continue;
    if (izproj[i] >= (1 << (settings_.nzbitsstub(0) - 1)))
      continue;

    //this is left from the original....
    if (iphiproj[i] >= (1 << settings_.nphibitsstub(5)) - 1)
      continue;
    if (iphiproj[i] <= 0)
      continue;

    if (rproj_[i] < settings_.rPS2S()) {
      iphiproj[i] >>= (settings_.nphibitsstub(5) - settings_.nphibitsstub(0));
    } else {
      izproj[i] >>= (settings_.nzbitsstub(0) - settings_.nzbitsstub(5));
    }

    if (rproj_[i] < settings_.rPS2S()) {
      if (iphider[i] < -(1 << (settings_.nbitsphiprojderL123() - 1)))
        iphider[i] = -(1 << (settings_.nbitsphiprojderL123() - 1));
      if (iphider[i] >= (1 << (settings_.nbitsphiprojderL123() - 1)))
        iphider[i] = (1 << (settings_.nbitsphiprojderL123() - 1)) - 1;
    } else {
      if (iphider[i] < -(1 << (settings_.nbitsphiprojderL456() - 1)))
        iphider[i] = -(1 << (settings_.nbitsphiprojderL456() - 1));
      if (iphider[i] >= (1 << (settings_.nbitsphiprojderL456() - 1)))
        iphider[i] = (1 << (settings_.nbitsphiprojderL456() - 1)) - 1;
    }

    projs[lproj_[i] - 1].init(settings_,
                              lproj_[i] - 1,
                              iphiproj[i],
                              izproj[i],
                              iphider[i],
                              izder[i],
                              phiproj[i],
                              zproj[i],
                              phider[i],
                              zder[i],
                              phiprojapprox[i],
                              zprojapprox[i],
                              phiderapprox[i],
                              zderapprox[i],
                              false);
  }

  if (std::abs(it * kt) > 1.0) {
    for (unsigned int i = 0; i < toZ_.size(); ++i) {
      iphiprojdisk[i] = phiprojdiskapprox[i] / kphiprojdisk;
      irprojdisk[i] = rprojdiskapprox[i] / krprojdisk;

      iphiderdisk[i] = phiderdiskapprox[i] / kphiderdisk;
      irderdisk[i] = rderdiskapprox[i] / krderdisk;

      //Check phi range of projection
      if (iphiprojdisk[i] <= 0)
        continue;
      if (iphiprojdisk[i] >= (1 << settings_.nphibitsstub(0)) - 1)
        continue;

      //Check r range of projection
      if (irprojdisk[i] < settings_.rmindisk() / krprojdisk || irprojdisk[i] >= settings_.rmaxdisk() / krprojdisk)
        continue;

      projs[N_LAYER + i + 1].init(settings_,
                                  N_LAYER + i + 1,
                                  iphiprojdisk[i],
                                  irprojdisk[i],
                                  iphiderdisk[i],
                                  irderdisk[i],
                                  phiprojdisk[i],
                                  rprojdisk[i],
                                  phiderdisk[i],
                                  rderdisk[i],
                                  phiprojdiskapprox[i],
                                  rprojdiskapprox[i],
                                  phiderdisk[i],
                                  rderdisk[i],
                                  false);
    }
  }

  if (settings_.writeMonitorData("TrackletPars")) {
    globals_->ofstream("trackletpars.txt")
        << layer_ << " , " << rinv << " , " << rinvapprox << " , " << phi0 << " , " << phi0approx << " , " << t << " , "
        << tapprox << " , " << z0 << " , " << z0approx << " , " << d0 << " , " << d0approx << endl;
  }

  Tracklet* tracklet = new Tracklet(settings_,
                                    iSeed_,
                                    innerFPGAStub,
                                    middleFPGAStub,
                                    outerFPGAStub,
                                    rinv,
                                    phi0,
                                    d0,
                                    z0,
                                    t,
                                    rinvapprox,
                                    phi0approx,
                                    d0approx,
                                    z0approx,
                                    tapprox,
                                    irinv,
                                    iphi0,
                                    id0,
                                    iz0,
                                    it,
                                    projs,
                                    false);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced " << getName()
                                 << " Found LLD tracklet in sector = " << iSector_ << " phi0 = " << phi0;

  tracklet->setTrackletIndex(trackletpars_->nTracklets());
  tracklet->setTCIndex(TCIndex_);

  if (settings_.writeMonitorData("Seeds")) {
    ofstream fout("seeds.txt", ofstream::app);
    fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " " << tracklet->getISeed() << endl;
    fout.close();
  }
  trackletpars_->addTracklet(tracklet);

  for (unsigned int j = 0; j < toR_.size(); j++) {
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "adding layer projection " << j << "/" << toR_.size() << " " << lproj_[j];
    if (tracklet->validProj(lproj_[j] - 1)) {
      addLayerProj(tracklet, lproj_[j]);
    }
  }

  for (unsigned int j = 0; j < toZ_.size(); j++) {
    int disk = dproj_[j];
    if (disk == 0)
      continue;
    if (it < 0)
      disk = -disk;
    if (settings_.debugTracklet())
      edm::LogVerbatim("Tracklet") << "adding disk projection " << j << "/" << toZ_.size() << " " << disk;
    if (tracklet->validProj(N_LAYER + abs(disk) - 1)) {
      addDiskProj(tracklet, disk);
    }
  }

  return true;
}

void TrackletCalculatorDisplaced::exactproj(double rproj,
                                            double rinv,
                                            double phi0,
                                            double d0,
                                            double t,
                                            double z0,
                                            double r0,
                                            double& phiproj,
                                            double& zproj,
                                            double& phider,
                                            double& zder) {
  double rho = 1 / rinv;
  if (rho < 0) {
    r0 = -r0;
  }
  phiproj = phi0 - asin((rproj * rproj + r0 * r0 - rho * rho) / (2 * rproj * r0));
  double beta = acos((rho * rho + r0 * r0 - rproj * rproj) / (2 * r0 * rho));
  zproj = z0 + t * std::abs(rho * beta);

  //not exact, but close
  phider = -0.5 * rinv / sqrt(1 - pow(0.5 * rproj * rinv, 2)) + d0 / (rproj * rproj);
  zder = t / sqrt(1 - pow(0.5 * rproj * rinv, 2));

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "exact proj layer at " << rproj << " : " << phiproj << " " << zproj;
}

void TrackletCalculatorDisplaced::exactprojdisk(double zproj,
                                                double rinv,
                                                double,
                                                double,  //phi0 and d0 are not used.
                                                double t,
                                                double z0,
                                                double x0,
                                                double y0,
                                                double& phiproj,
                                                double& rproj,
                                                double& phider,
                                                double& rder) {
  //protect against t=0
  if (std::abs(t) < 0.1)
    t = 0.1;
  if (t < 0)
    zproj = -zproj;
  double rho = std::abs(1 / rinv);
  double beta = (zproj - z0) / (t * rho);
  double phiV = atan2(-y0, -x0);
  double c = rinv > 0 ? -1 : 1;

  double x = x0 + rho * cos(phiV + c * beta);
  double y = y0 + rho * sin(phiV + c * beta);

  phiproj = atan2(y, x);

  phiproj = reco::reduceRange(phiproj - phimin_);

  rproj = sqrt(x * x + y * y);

  phider = c / t / (x * x + y * y) * (rho + x0 * cos(phiV + c * beta) + y0 * sin(phiV + c * beta));
  rder = c / t / rproj * (y0 * cos(phiV + c * beta) - x0 * sin(phiV + c * beta));

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "exact proj disk at" << zproj << " : " << phiproj << " " << rproj;
}

void TrackletCalculatorDisplaced::exacttracklet(double r1,
                                                double z1,
                                                double phi1,
                                                double r2,
                                                double z2,
                                                double phi2,
                                                double r3,
                                                double z3,
                                                double phi3,
                                                int take3,
                                                double& rinv,
                                                double& phi0,
                                                double& d0,
                                                double& t,
                                                double& z0,
                                                double phiproj[N_LAYER - 2],
                                                double zproj[N_LAYER - 2],
                                                double phiprojdisk[N_DISK],
                                                double rprojdisk[N_DISK],
                                                double phider[N_LAYER - 2],
                                                double zder[N_LAYER - 2],
                                                double phiderdisk[N_DISK],
                                                double rderdisk[N_DISK]) {
  //two lines perpendicular to the 1->2 and 2->3
  double x1 = r1 * cos(phi1);
  double x2 = r2 * cos(phi2);
  double x3 = r3 * cos(phi3);

  double y1 = r1 * sin(phi1);
  double y2 = r2 * sin(phi2);
  double y3 = r3 * sin(phi3);

  double dy21 = y2 - y1;
  double dy32 = y3 - y2;

  //Hack to protect against dividing by zero
  //code should be rewritten to avoid this
  if (dy21 == 0.0)
    dy21 = 1e-9;
  if (dy32 == 0.0)
    dy32 = 1e-9;

  double k1 = -(x2 - x1) / dy21;
  double k2 = -(x3 - x2) / dy32;
  double b1 = 0.5 * (y2 + y1) - 0.5 * (x1 + x2) * k1;
  double b2 = 0.5 * (y3 + y2) - 0.5 * (x2 + x3) * k2;
  //their intersection gives the center of the circle
  double y0 = (b1 * k2 - b2 * k1) / (k2 - k1);
  double x0 = (b1 - b2) / (k2 - k1);
  //get the radius three ways:
  double R1 = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2));
  double R2 = sqrt(pow(x2 - x0, 2) + pow(y2 - y0, 2));
  double R3 = sqrt(pow(x3 - x0, 2) + pow(y3 - y0, 2));
  //check if the same
  double eps1 = std::abs(R1 / R2 - 1);
  double eps2 = std::abs(R3 / R2 - 1);
  if (eps1 > 1e-10 || eps2 > 1e-10)
    edm::LogVerbatim("Tracklet") << "&&&&&&&&&&&& bad circle! " << R1 << "\t" << R2 << "\t" << R3;

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "phimin_: " << phimin_ << " phimax_: " << phimax_;
  //results
  rinv = 1. / R1;
  phi0 = 0.5 * M_PI + atan2(y0, x0);

  phi0 -= phimin_;

  d0 = -R1 + sqrt(x0 * x0 + y0 * y0);
  //sign of rinv:
  double dphi = reco::reduceRange(phi3 - atan2(y0, x0));
  if (dphi < 0) {
    rinv = -rinv;
    d0 = -d0;
    phi0 = phi0 + M_PI;
  }
  phi0 = angle0to2pi::make0To2pi(phi0);

  //now in RZ:
  //turning angle
  double beta1 = reco::reduceRange(atan2(y1 - y0, x1 - x0) - atan2(-y0, -x0));
  double beta2 = reco::reduceRange(atan2(y2 - y0, x2 - x0) - atan2(-y0, -x0));
  double beta3 = reco::reduceRange(atan2(y3 - y0, x3 - x0) - atan2(-y0, -x0));

  double t12 = (z2 - z1) / std::abs(beta2 - beta1) / R1;
  double z12 = (z1 * beta2 - z2 * beta1) / (beta2 - beta1);
  double t13 = (z3 - z1) / std::abs(beta3 - beta1) / R1;
  double z13 = (z1 * beta3 - z3 * beta1) / (beta3 - beta1);

  if (take3 > 0) {
    //take 13 (large lever arm)
    t = t13;
    z0 = z13;
  } else {
    //take 12 (pixel layers)
    t = t12;
    z0 = z12;
  }

  for (unsigned int i = 0; i < toR_.size(); i++) {
    exactproj(toR_[i], rinv, phi0, d0, t, z0, sqrt(x0 * x0 + y0 * y0), phiproj[i], zproj[i], phider[i], zder[i]);
  }

  for (unsigned int i = 0; i < toZ_.size(); i++) {
    exactprojdisk(toZ_[i], rinv, phi0, d0, t, z0, x0, y0, phiprojdisk[i], rprojdisk[i], phiderdisk[i], rderdisk[i]);
  }

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "exact tracklet: " << rinv << " " << phi0 << " " << t << " " << z0 << " " << d0;
}

void TrackletCalculatorDisplaced::approxproj(double halfRinv,
                                             double phi0,
                                             double d0,
                                             double t,
                                             double z0,
                                             double halfRinv_0,
                                             double d0_0,  // zeroeth order result for higher order terms calculation
                                             double rmean,
                                             double& phiproj,
                                             double& phiprojder,
                                             double& zproj,
                                             double& zprojder) {
  if (std::abs(2.0 * halfRinv) > settings_.rinvcut() || std::abs(z0) > settings_.disp_z0cut() ||
      std::abs(d0) > settings_.maxd0()) {
    phiproj = 0.0;
    return;
  }
  double rmeanInv = 1.0 / rmean;

  phiproj = phi0 + rmean * (-halfRinv + 2.0 * d0_0 * halfRinv_0 * halfRinv_0) +
            rmeanInv * (-d0 + halfRinv_0 * d0_0 * d0_0) + sixth * pow(-rmean * halfRinv_0 - rmeanInv * d0_0, 3);
  phiprojder = -halfRinv + d0 * rmeanInv * rmeanInv;  //removed all high terms

  zproj = z0 + t * rmean - 0.5 * rmeanInv * t * d0_0 * d0_0 - t * rmean * halfRinv * d0 +
          sixth * pow(rmean, 3) * t * halfRinv_0 * halfRinv_0;
  zprojder = t;  // removed all high terms

  phiproj = angle0to2pi::make0To2pi(phiproj);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "approx proj layer at " << rmean << " : " << phiproj << " " << zproj << endl;
}

void TrackletCalculatorDisplaced::approxprojdisk(double halfRinv,
                                                 double phi0,
                                                 double d0,
                                                 double t,
                                                 double z0,
                                                 double halfRinv_0,
                                                 double d0_0,  // zeroeth order result for higher order terms calculation
                                                 double zmean,
                                                 double& phiproj,
                                                 double& phiprojder,
                                                 double& rproj,
                                                 double& rprojder) {
  if (std::abs(2.0 * halfRinv) > settings_.rinvcut() || std::abs(z0) > settings_.disp_z0cut() ||
      std::abs(d0) > settings_.maxd0()) {
    phiproj = 0.0;
    return;
  }

  if (t < 0)
    zmean = -zmean;

  double zmeanInv = 1.0 / zmean, rstar = (zmean - z0) / t,
         epsilon = 0.5 * zmeanInv * zmeanInv * d0_0 * d0_0 * t * t + halfRinv * d0 -
                   sixth * rstar * rstar * halfRinv_0 * halfRinv_0;

  rproj = rstar * (1 + epsilon);
  rprojder = 1 / t;

  double A = rproj * halfRinv;
  double B = -d0 * t * zmeanInv * (1 + z0 * zmeanInv) * (1 - epsilon);
  double C = -d0 * halfRinv;
  double A_0 = rproj * halfRinv_0;
  double B_0 = -d0_0 * t * zmeanInv * (1 + z0 * zmeanInv) * (1 - epsilon);
  // double C_0 = -d0_0 * halfRinv_0;

  phiproj = phi0 - A + B * (1 + C - 2 * A_0 * A_0) + sixth * pow(-A_0 + B_0, 3);
  phiprojder = -halfRinv / t + d0 * t * zmeanInv * zmeanInv;

  phiproj = angle0to2pi::make0To2pi(phiproj);

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "approx proj disk at" << zmean << " : " << phiproj << " " << rproj << endl;
}

void TrackletCalculatorDisplaced::approxtracklet(double r1,
                                                 double z1,
                                                 double phi1,
                                                 double r2,
                                                 double z2,
                                                 double phi2,
                                                 double r3,
                                                 double z3,
                                                 double phi3,
                                                 bool take3,
                                                 unsigned ndisks,
                                                 double& rinv,
                                                 double& phi0,
                                                 double& d0,
                                                 double& t,
                                                 double& z0,
                                                 double phiproj[4],
                                                 double zproj[4],
                                                 double phider[4],
                                                 double zder[4],
                                                 double phiprojdisk[5],
                                                 double rprojdisk[5],
                                                 double phiderdisk[5],
                                                 double rderdisk[5]) {
  double a = 1.0 / ((r1 - r2) * (r1 - r3));
  double b = 1.0 / ((r1 - r2) * (r2 - r3));
  double c = 1.0 / ((r1 - r3) * (r2 - r3));

  // first iteration in r-phi plane
  double halfRinv_0 = -phi1 * r1 * a + phi2 * r2 * b - phi3 * r3 * c;
  double d0_0 = r1 * r2 * r3 * (-phi1 * a + phi2 * b - phi3 * c);

  // corrections to phi1, phi2, and phi3
  double r = r2, z = z2;
  if (take3)
    r = r3, z = z3;

  double d0OverR1 = d0_0 * rzmeanInv_[0] * (ndisks > 2 ? std::abs((z - z1) / (r - r1)) : 1.0);
  double d0OverR2 = d0_0 * rzmeanInv_[1] * (ndisks > 1 ? std::abs((z - z1) / (r - r1)) : 1.0);
  double d0OverR3 = d0_0 * rzmeanInv_[2] * (ndisks > 0 ? std::abs((z - z1) / (r - r1)) : 1.0);

  double d0OverR = d0OverR2;
  if (take3)
    d0OverR = d0OverR3;

  double c1 = d0_0 * halfRinv_0 * d0OverR1 + 2.0 * d0_0 * halfRinv_0 * r1 * halfRinv_0 +
              sixth * pow(-r1 * halfRinv_0 - d0OverR1, 3);
  double c2 = d0_0 * halfRinv_0 * d0OverR2 + 2.0 * d0_0 * halfRinv_0 * r2 * halfRinv_0 +
              sixth * pow(-r2 * halfRinv_0 - d0OverR2, 3);
  double c3 = d0_0 * halfRinv_0 * d0OverR3 + 2.0 * d0_0 * halfRinv_0 * r3 * halfRinv_0 +
              sixth * pow(-r3 * halfRinv_0 - d0OverR3, 3);

  double phi1c = phi1 - c1;
  double phi2c = phi2 - c2;
  double phi3c = phi3 - c3;

  // second iteration in r-phi plane
  double halfRinv = -phi1c * r1 * a + phi2c * r2 * b - phi3c * r3 * c;
  phi0 = -phi1c * r1 * (r2 + r3) * a + phi2c * r2 * (r1 + r3) * b - phi3c * r3 * (r1 + r2) * c;
  d0 = r1 * r2 * r3 * (-phi1c * a + phi2c * b - phi3c * c);

  t = ((z - z1) / (r - r1)) *
      (1. + d0 * halfRinv - 0.5 * d0OverR1 * d0OverR - sixth * (r1 * r1 + r2 * r2 + r1 * r2) * halfRinv_0 * halfRinv_0);
  z0 = z1 - t * r1 * (1.0 - d0_0 * halfRinv_0 - 0.5 * d0OverR1 * d0OverR1 + sixth * r1 * r1 * halfRinv_0 * halfRinv_0);

  rinv = 2.0 * halfRinv;
  phi0 += -phimin_;

  phi0 = angle0to2pi::make0To2pi(phi0);

  for (unsigned int i = 0; i < toR_.size(); i++) {
    approxproj(halfRinv,
               phi0,
               d0,
               t,
               z0,
               halfRinv_0,
               d0_0,  // added _0 version for high term calculations
               toR_.at(i),
               phiproj[i],
               phider[i],
               zproj[i],
               zder[i]);
  }

  for (unsigned int i = 0; i < toZ_.size(); i++) {
    approxprojdisk(halfRinv,
                   phi0,
                   d0,
                   t,
                   z0,
                   halfRinv_0,
                   d0_0,  // added _0 version for high term calculations
                   toZ_.at(i),
                   phiprojdisk[i],
                   phiderdisk[i],
                   rprojdisk[i],
                   rderdisk[i]);
  }

  if (std::abs(rinv) > settings_.rinvcut() || std::abs(z0) > settings_.disp_z0cut() ||
      std::abs(d0) > settings_.maxd0()) {
    phi0 = 0.0;
    return;
  }

  if (settings_.debugTracklet())
    edm::LogVerbatim("Tracklet") << "TCD approx tracklet: " << rinv << " " << phi0 << " " << t << " " << z0 << " " << d0
                                 << endl;
}

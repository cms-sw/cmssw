#include "L1Trigger/TrackFindingTracklet/interface/FitTrack.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackDerTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/HybridFit.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
using namespace trklet;

FitTrack::FitTrack(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global), trackfit_(nullptr) {}

void FitTrack::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "trackout") {
    TrackFitMemory* tmp = dynamic_cast<TrackFitMemory*>(memory);
    assert(tmp != nullptr);
    trackfit_ = tmp;
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " addOutput, output = " << output << " not known";
}

void FitTrack::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input.substr(0, 4) == "tpar") {
    auto* tmp = dynamic_cast<TrackletParametersMemory*>(memory);
    assert(tmp != nullptr);
    seedtracklet_.push_back(tmp);
    return;
  }
  if (input.substr(0, 10) == "fullmatch1") {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    fullmatch1_.push_back(tmp);
    return;
  }
  if (input.substr(0, 10) == "fullmatch2") {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    fullmatch2_.push_back(tmp);
    return;
  }
  if (input.substr(0, 10) == "fullmatch3") {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    fullmatch3_.push_back(tmp);
    return;
  }
  if (input.substr(0, 10) == "fullmatch4") {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    fullmatch4_.push_back(tmp);
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " input = " << input << " not found";
}

#ifdef USEHYBRID
void FitTrack::trackFitKF(Tracklet* tracklet,
                          std::vector<const Stub*>& trackstublist,
                          std::vector<std::pair<int, int>>& stubidslist) {
  if (settings_.doKF()) {
    // From full match lists, collect all the stubs associated with the tracklet seed

    // Get seed stubs first
    trackstublist.emplace_back(tracklet->innerFPGAStub());
    if (tracklet->getISeed() >= (int)N_TRKLSEED + 1)
      trackstublist.emplace_back(tracklet->middleFPGAStub());
    trackstublist.emplace_back(tracklet->outerFPGAStub());

    // Now get ALL matches (can have multiple per layer)
    for (const auto& i : fullmatch1_) {
      for (unsigned int j = 0; j < i->nMatches(); j++) {
        if (i->getTracklet(j)->TCID() == tracklet->TCID()) {
          trackstublist.push_back(i->getMatch(j).second);
        }
      }
    }

    for (const auto& i : fullmatch2_) {
      for (unsigned int j = 0; j < i->nMatches(); j++) {
        if (i->getTracklet(j)->TCID() == tracklet->TCID()) {
          trackstublist.push_back(i->getMatch(j).second);
        }
      }
    }

    for (const auto& i : fullmatch3_) {
      for (unsigned int j = 0; j < i->nMatches(); j++) {
        if (i->getTracklet(j)->TCID() == tracklet->TCID()) {
          trackstublist.push_back(i->getMatch(j).second);
        }
      }
    }

    for (const auto& i : fullmatch4_) {
      for (unsigned int j = 0; j < i->nMatches(); j++) {
        if (i->getTracklet(j)->TCID() == tracklet->TCID()) {
          trackstublist.push_back(i->getMatch(j).second);
        }
      }
    }

    // For merge removal, loop through the resulting list of stubs to calculate their stubids
    if (settings_.removalType() == "merge") {
      for (const auto& it : trackstublist) {
        int layer = it->layer().value() + 1;  // Assume layer (1-6) stub first
        if (it->layer().value() < 0) {        // if disk stub, though...
          layer = it->disk().value() + 10 * it->disk().value() / abs(it->disk().value());  //disk = +/- 11-15
        }
        stubidslist.push_back(std::make_pair(layer, it->phiregionaddress()));
      }

      // And that's all we need! The rest is just for fitting (in PurgeDuplicate)
      return;
    }

    HybridFit hybridFitter(iSector_, settings_, globals_);
    hybridFitter.Fit(tracklet, trackstublist);
    return;
  }
}
#endif

void FitTrack::trackFitChisq(Tracklet* tracklet, std::vector<const Stub*>&, std::vector<std::pair<int, int>>&) {
  if (globals_->trackDerTable() == nullptr) {
    TrackDerTable* derTablePtr = new TrackDerTable(settings_);

    derTablePtr->readPatternFile(settings_.fitPatternFile());
    derTablePtr->fillTable();
    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "Number of entries in derivative table: " << derTablePtr->getEntries();
    }
    assert(derTablePtr->getEntries() != 0);

    globals_->trackDerTable() = derTablePtr;
  }

  const TrackDerTable& derTable = *globals_->trackDerTable();

  //First step is to build list of layers and disks.
  int layers[N_LAYER];
  double r[N_LAYER];
  unsigned int nlayers = 0;  // layers with found stub-projections
  int disks[N_DISK];
  double z[N_DISK];
  unsigned int ndisks = 0;  // disks with found stub-projections

  // residuals for each stub
  double phiresid[N_FITSTUB];
  double zresid[N_FITSTUB];
  double phiresidexact[N_FITSTUB];
  double zresidexact[N_FITSTUB];
  int iphiresid[N_FITSTUB];
  int izresid[N_FITSTUB];
  double alpha[N_FITSTUB];

  for (unsigned int i = 0; i < N_FITSTUB; i++) {
    iphiresid[i] = 0;
    izresid[i] = 0;
    alpha[i] = 0.0;

    phiresid[i] = 0.0;
    zresid[i] = 0.0;
    phiresidexact[i] = 0.0;
    zresidexact[i] = 0.0;
  }

  std::bitset<N_LAYER> lmatches;     //layer matches
  std::bitset<N_DISK * 2> dmatches;  //disk matches (2 per disk to separate 2S from PS)

  int mult = 1;

  unsigned int layermask = 0;
  unsigned int diskmask = 0;
  unsigned int alphaindex = 0;
  unsigned int power = 1;

  double t = tracklet->t();
  double rinv = tracklet->rinv();

  if (tracklet->isBarrel()) {
    for (unsigned int l = 1; l <= N_LAYER; l++) {
      if (l == (unsigned int)tracklet->layer() || l == (unsigned int)tracklet->layer() + 1) {
        lmatches.set(N_LAYER - l);
        layermask |= (1 << (N_LAYER - l));
        layers[nlayers++] = l;
        continue;
      }
      if (tracklet->match(l - 1)) {
        const Residual& resid = tracklet->resid(l - 1);
        lmatches.set(N_LAYER - l);
        layermask |= (1 << (N_LAYER - l));
        phiresid[nlayers] = resid.phiresidapprox();
        zresid[nlayers] = resid.rzresidapprox();
        phiresidexact[nlayers] = resid.phiresid();
        zresidexact[nlayers] = resid.rzresid();
        iphiresid[nlayers] = resid.fpgaphiresid().value();
        izresid[nlayers] = resid.fpgarzresid().value();

        layers[nlayers++] = l;
      }
    }

    for (unsigned int d = 1; d <= N_DISK; d++) {
      if (layermask & (1 << (d - 1)))
        continue;

      if (mult == 1 << (3 * settings_.alphaBitsTable()))
        continue;

      if (ndisks + nlayers >= N_FITSTUB)
        continue;
      if (tracklet->match(N_LAYER + d - 1)) {
        const Residual& resid = tracklet->resid(N_LAYER + d - 1);
        double pitch = settings_.stripPitch(resid.stubptr()->l1tstub()->isPSmodule());
        if (std::abs(resid.stubptr()->l1tstub()->alpha(pitch)) < 1e-20) {
          dmatches.set(2 * d - 1);
          diskmask |= (1 << (2 * (N_DISK - d) + 1));
        } else {
          int ialpha = resid.stubptr()->alpha().value();
          int nalpha = resid.stubptr()->alpha().nbits();
          nalpha = nalpha - settings_.alphaBitsTable();
          ialpha = (1 << (settings_.alphaBitsTable() - 1)) + (ialpha >> nalpha);

          alphaindex += ialpha * power;
          power = power << settings_.alphaBitsTable();
          dmatches.set(2 * (N_DISK - d));
          diskmask |= (1 << (2 * (N_DISK - d)));
          mult = mult << settings_.alphaBitsTable();
        }
        alpha[ndisks] = resid.stubptr()->l1tstub()->alpha(pitch);
        phiresid[nlayers + ndisks] = resid.phiresidapprox();
        zresid[nlayers + ndisks] = resid.rzresidapprox();
        phiresidexact[nlayers + ndisks] = resid.phiresid();
        zresidexact[nlayers + ndisks] = resid.rzresid();
        iphiresid[nlayers + ndisks] = resid.fpgaphiresid().value();
        izresid[nlayers + ndisks] = resid.fpgarzresid().value();

        disks[ndisks++] = d;
      }
    }

    if (settings_.writeMonitorData("HitPattern")) {
      if (mult <= 1 << (3 * settings_.alphaBitsTable())) {
        globals_->ofstream("hitpattern.txt")
            << lmatches.to_string() << " " << dmatches.to_string() << " " << mult << endl;
      }
    }
  }

  if (tracklet->isDisk()) {
    for (unsigned int l = 1; l <= 2; l++) {
      if (tracklet->match(l - 1)) {
        lmatches.set(N_LAYER - l);

        layermask |= (1 << (N_LAYER - l));
        const Residual& resid = tracklet->resid(l - 1);
        phiresid[nlayers] = resid.phiresidapprox();
        zresid[nlayers] = resid.rzresidapprox();
        phiresidexact[nlayers] = resid.phiresid();
        zresidexact[nlayers] = resid.rzresid();
        iphiresid[nlayers] = resid.fpgaphiresid().value();
        izresid[nlayers] = resid.fpgarzresid().value();

        layers[nlayers++] = l;
      }
    }

    for (int d1 = 1; d1 <= N_DISK; d1++) {
      int d = d1;

      // skip F/B5 if there's already a L2 match
      if (d == 5 and layermask & (1 << 4))
        continue;

      if (tracklet->fpgat().value() < 0.0)
        d = -d1;
      if (d1 == abs(tracklet->disk()) || d1 == abs(tracklet->disk()) + 1) {
        dmatches.set(2 * d1 - 1);
        diskmask |= (1 << (2 * (N_DISK - d1) + 1));
        alpha[ndisks] = 0.0;
        disks[ndisks++] = d;
        continue;
      }

      if (ndisks + nlayers >= N_FITSTUB)
        continue;
      if (tracklet->match(N_LAYER + abs(d) - 1)) {
        const Residual& resid = tracklet->resid(N_LAYER + abs(d) - 1);
        double pitch = settings_.stripPitch(resid.stubptr()->l1tstub()->isPSmodule());
        if (std::abs(resid.stubptr()->l1tstub()->alpha(pitch)) < 1e-20) {
          dmatches.set(2 * d1 - 1);
          diskmask |= (1 << (2 * (N_DISK - d1) + 1));
        } else {
          int ialpha = resid.stubptr()->alpha().value();
          int nalpha = resid.stubptr()->alpha().nbits();
          nalpha = nalpha - settings_.alphaBitsTable();
          ialpha = (1 << (settings_.alphaBitsTable() - 1)) + (ialpha >> nalpha);

          alphaindex += ialpha * power;
          power = power << settings_.alphaBitsTable();
          dmatches.set(2 * (N_DISK - d1));
          diskmask |= (1 << (2 * (N_DISK - d1)));
          mult = mult << settings_.alphaBitsTable();
        }

        alpha[ndisks] = resid.stubptr()->l1tstub()->alpha(pitch);
        assert(std::abs(resid.phiresidapprox()) < 0.2);
        phiresid[nlayers + ndisks] = resid.phiresidapprox();
        zresid[nlayers + ndisks] = resid.rzresidapprox();
        assert(std::abs(resid.phiresid()) < 0.2);
        phiresidexact[nlayers + ndisks] = resid.phiresid();
        zresidexact[nlayers + ndisks] = resid.rzresid();
        iphiresid[nlayers + ndisks] = resid.fpgaphiresid().value();
        izresid[nlayers + ndisks] = resid.fpgarzresid().value();

        disks[ndisks++] = d;
      }
    }
  }

  if (tracklet->isOverlap()) {
    for (unsigned int l = 1; l <= 2; l++) {
      if (l == (unsigned int)tracklet->layer()) {
        lmatches.set(N_LAYER - l);
        layermask |= (1 << (N_LAYER - l));
        layers[nlayers++] = l;
        continue;
      }
      if (tracklet->match(l - 1)) {
        lmatches.set(N_LAYER - l);
        layermask |= (1 << (N_LAYER - l));
        const Residual& resid = tracklet->resid(l - 1);
        assert(std::abs(resid.phiresidapprox()) < 0.2);
        phiresid[nlayers] = resid.phiresidapprox();
        zresid[nlayers] = resid.rzresidapprox();
        assert(std::abs(resid.phiresid()) < 0.2);
        phiresidexact[nlayers] = resid.phiresid();
        zresidexact[nlayers] = resid.rzresid();
        iphiresid[nlayers] = resid.fpgaphiresid().value();
        izresid[nlayers] = resid.fpgarzresid().value();

        layers[nlayers++] = l;
      }
    }

    for (unsigned int d1 = 1; d1 <= N_DISK; d1++) {
      if (mult == 1 << (3 * settings_.alphaBitsTable()))
        continue;
      int d = d1;
      if (tracklet->fpgat().value() < 0.0)
        d = -d1;
      if (d == tracklet->disk()) {  //All seeds in PS modules
        disks[ndisks] = tracklet->disk();
        dmatches.set(2 * d1 - 1);
        diskmask |= (1 << (2 * (N_DISK - d1) + 1));
        ndisks++;
        continue;
      }

      if (ndisks + nlayers >= N_FITSTUB)
        continue;
      if (tracklet->match(N_LAYER + abs(d) - 1)) {
        const Residual& resid = tracklet->resid(N_LAYER + abs(d) - 1);
        double pitch = settings_.stripPitch(resid.stubptr()->l1tstub()->isPSmodule());
        if (std::abs(resid.stubptr()->l1tstub()->alpha(pitch)) < 1e-20) {
          dmatches.set(2 * (N_DISK - d1));
          diskmask |= (1 << (2 * (N_DISK - d1) + 1));
          FPGAWord tmp;
          tmp.set(diskmask, 10);
        } else {
          int ialpha = resid.stubptr()->alpha().value();
          int nalpha = resid.stubptr()->alpha().nbits();
          nalpha = nalpha - settings_.alphaBitsTable();
          ialpha = (1 << (settings_.alphaBitsTable() - 1)) + (ialpha >> nalpha);

          alphaindex += ialpha * power;
          power = power << settings_.alphaBitsTable();
          dmatches.set(2 * (N_DISK - d1));
          diskmask |= (1 << (2 * (N_DISK - d1)));
          FPGAWord tmp;
          tmp.set(diskmask, 10);
          mult = mult << settings_.alphaBitsTable();
        }

        alpha[ndisks] = resid.stubptr()->l1tstub()->alpha(pitch);
        assert(std::abs(resid.phiresidapprox()) < 0.2);
        phiresid[nlayers + ndisks] = resid.phiresidapprox();
        zresid[nlayers + ndisks] = resid.rzresidapprox();
        assert(std::abs(resid.phiresid()) < 0.2);
        phiresidexact[nlayers + ndisks] = resid.phiresid();
        zresidexact[nlayers + ndisks] = resid.rzresid();
        iphiresid[nlayers + ndisks] = resid.fpgaphiresid().value();
        izresid[nlayers + ndisks] = resid.fpgarzresid().value();

        disks[ndisks++] = d;
      }
    }
  }

  int rinvindex =
      (1 << (settings_.nrinvBitsTable() - 1)) * rinv / settings_.rinvmax() + (1 << (settings_.nrinvBitsTable() - 1));
  if (rinvindex < 0)
    rinvindex = 0;
  if (rinvindex >= (1 << settings_.nrinvBitsTable()))
    rinvindex = (1 << settings_.nrinvBitsTable()) - 1;

  const TrackDer* derivatives = derTable.getDerivatives(layermask, diskmask, alphaindex, rinvindex);

  if (derivatives == nullptr) {
    if (settings_.warnNoDer()) {
      FPGAWord tmpl, tmpd;
      tmpl.set(layermask, 6);
      tmpd.set(diskmask, 10);
      edm::LogVerbatim("Tracklet") << "No derivative for layermask, diskmask : " << layermask << " " << tmpl.str()
                                   << " " << diskmask << " " << tmpd.str() << " eta = " << asinh(t);
    }
    return;
  }

  double ttabi = TrackDerTable::tpar(settings_, diskmask, layermask);
  if (t < 0.0)
    ttabi = -ttabi;
  double ttab = ttabi;

  if (settings_.debugTracklet()) {
    edm::LogVerbatim("Tracklet") << "Doing trackfit in  " << getName();
  }

  int sign = 1;
  if (t < 0.0)
    sign = -1;

  double rstub[6];

  double realrstub[3];
  realrstub[0] = -1.0;
  realrstub[1] = -1.0;
  realrstub[2] = -1.0;

  for (unsigned i = 0; i < nlayers; i++) {
    r[i] = settings_.rmean(layers[i] - 1);
    if (layers[i] == tracklet->layer()) {
      if (tracklet->isOverlap()) {
        realrstub[i] = tracklet->outerFPGAStub()->l1tstub()->r();
      } else {
        realrstub[i] = tracklet->innerFPGAStub()->l1tstub()->r();
      }
    }
    if (layers[i] == tracklet->layer() + 1) {
      realrstub[i] = tracklet->outerFPGAStub()->l1tstub()->r();
    }
    if (tracklet->match(layers[i] - 1) && layers[i] < 4) {
      const Stub* stubptr = tracklet->resid(layers[i] - 1).stubptr();
      realrstub[i] = stubptr->l1tstub()->r();
      assert(std::abs(realrstub[i] - r[i]) < 5.0);
    }
    rstub[i] = r[i];
  }
  for (unsigned i = 0; i < ndisks; i++) {
    z[i] = sign * settings_.zmean(abs(disks[i]) - 1);
    rstub[i + nlayers] = z[i] / ttabi;
  }

  double D[N_FITPARAM][N_FITSTUB * 2];
  double MinvDt[N_FITPARAM][N_FITSTUB * 2];
  int iD[N_FITPARAM][N_FITSTUB * 2];
  int iMinvDt[N_FITPARAM][N_FITSTUB * 2];
  double sigma[N_FITSTUB * 2];
  double kfactor[N_FITSTUB * 2];

  unsigned int n = nlayers + ndisks;

  if (settings_.exactderivatives()) {
    TrackDerTable::calculateDerivatives(
        settings_, nlayers, r, ndisks, z, alpha, t, rinv, D, iD, MinvDt, iMinvDt, sigma, kfactor);
    ttabi = t;
    ttab = t;
  } else {
    if (settings_.exactderivativesforfloating()) {
      TrackDerTable::calculateDerivatives(
          settings_, nlayers, r, ndisks, z, alpha, t, rinv, D, iD, MinvDt, iMinvDt, sigma, kfactor);

      double MinvDtDummy[N_FITPARAM][N_FITSTUB * 2];
      derivatives->fill(tracklet->fpgat().value(), MinvDtDummy, iMinvDt);
      ttab = t;
    } else {
      derivatives->fill(tracklet->fpgat().value(), MinvDt, iMinvDt);
    }
  }

  if (!settings_.exactderivatives()) {
    for (unsigned int i = 0; i < nlayers; i++) {
      if (r[i] > settings_.rPS2S())
        continue;
      for (unsigned int ii = 0; ii < nlayers; ii++) {
        if (r[ii] > settings_.rPS2S())
          continue;

        double tder = derivatives->tdzcorr(i, ii);
        double zder = derivatives->z0dzcorr(i, ii);

        double dr = realrstub[i] - r[i];

        MinvDt[2][2 * ii + 1] += dr * tder;
        MinvDt[3][2 * ii + 1] += dr * zder;

        int itder = derivatives->itdzcorr(i, ii);
        int izder = derivatives->iz0dzcorr(i, ii);

        int idr = dr / settings_.kr();

        iMinvDt[2][2 * ii + 1] += ((idr * itder) >> settings_.rcorrbits());
        iMinvDt[3][2 * ii + 1] += ((idr * izder) >> settings_.rcorrbits());
      }
    }
  }

  double rinvseed = tracklet->rinvapprox();
  double phi0seed = tracklet->phi0approx();
  double tseed = tracklet->tapprox();
  double z0seed = tracklet->z0approx();

  double rinvseedexact = tracklet->rinv();
  double phi0seedexact = tracklet->phi0();
  double tseedexact = tracklet->t();
  double z0seedexact = tracklet->z0();

  double chisqseed = 0.0;
  double chisqseedexact = 0.0;

  double delta[2 * N_FITSTUB];
  double deltaexact[2 * N_FITSTUB];
  int idelta[2 * N_FITSTUB];

  for (unsigned int i = 0; i < 2 * N_FITSTUB; i++) {
    delta[i] = 0.0;
    deltaexact[i] = 0.0;
    idelta[i] = 0;
  }

  int j = 0;

  for (unsigned int i = 0; i < n; i++) {
    if (i >= nlayers) {
      iphiresid[i] *= (t / ttabi);
      phiresid[i] *= (t / ttab);
      phiresidexact[i] *= (t / ttab);
    }

    idelta[j] = iphiresid[i];
    delta[j] = phiresid[i];
    if (std::abs(phiresid[i]) > 0.2) {
      edm::LogWarning("Tracklet") << getName() << " WARNING too large phiresid: " << phiresid[i] << " "
                                  << phiresidexact[i];
    }
    assert(std::abs(phiresid[i]) < 1.0);
    assert(std::abs(phiresidexact[i]) < 1.0);
    deltaexact[j++] = phiresidexact[i];

    idelta[j] = izresid[i];
    delta[j] = zresid[i];
    deltaexact[j++] = zresidexact[i];

    chisqseed += (delta[j - 2] * delta[j - 2] + delta[j - 1] * delta[j - 1]);
    chisqseedexact += (deltaexact[j - 2] * deltaexact[j - 2] + deltaexact[j - 1] * deltaexact[j - 1]);
  }
  assert(j <= 12);

  double drinv = 0.0;
  double dphi0 = 0.0;
  double dt = 0.0;
  double dz0 = 0.0;

  double drinvexact = 0.0;
  double dphi0exact = 0.0;
  double dtexact = 0.0;
  double dz0exact = 0.0;

  int idrinv = 0;
  int idphi0 = 0;
  int idt = 0;
  int idz0 = 0;

  double drinv_cov = 0.0;
  double dphi0_cov = 0.0;
  double dt_cov = 0.0;
  double dz0_cov = 0.0;

  double drinv_covexact = 0.0;
  double dphi0_covexact = 0.0;
  double dt_covexact = 0.0;
  double dz0_covexact = 0.0;

  for (unsigned int j = 0; j < 2 * n; j++) {
    drinv -= MinvDt[0][j] * delta[j];
    dphi0 -= MinvDt[1][j] * delta[j];
    dt -= MinvDt[2][j] * delta[j];
    dz0 -= MinvDt[3][j] * delta[j];

    drinv_cov += D[0][j] * delta[j];
    dphi0_cov += D[1][j] * delta[j];
    dt_cov += D[2][j] * delta[j];
    dz0_cov += D[3][j] * delta[j];

    drinvexact -= MinvDt[0][j] * deltaexact[j];
    dphi0exact -= MinvDt[1][j] * deltaexact[j];
    dtexact -= MinvDt[2][j] * deltaexact[j];
    dz0exact -= MinvDt[3][j] * deltaexact[j];

    drinv_covexact += D[0][j] * deltaexact[j];
    dphi0_covexact += D[1][j] * deltaexact[j];
    dt_covexact += D[2][j] * deltaexact[j];
    dz0_covexact += D[3][j] * deltaexact[j];

    idrinv += ((iMinvDt[0][j] * idelta[j]));
    idphi0 += ((iMinvDt[1][j] * idelta[j]));
    idt += ((iMinvDt[2][j] * idelta[j]));
    idz0 += ((iMinvDt[3][j] * idelta[j]));

    if (false && j % 2 == 0) {
      edm::LogVerbatim("Tracklet") << "DEBUG CHI2FIT " << j << " " << rinvseed << " + " << MinvDt[0][j] * delta[j]
                                   << " " << MinvDt[0][j] << " " << delta[j] * rstub[j / 2] * 10000 << " \n"
                                   << j << " " << tracklet->fpgarinv().value() * settings_.krinvpars() << " + "
                                   << ((iMinvDt[0][j] * idelta[j])) * settings_.krinvpars() / 1024.0 << " "
                                   << iMinvDt[0][j] * settings_.krinvpars() / settings_.kphi() / 1024.0 << " "
                                   << idelta[j] * settings_.kphi() * rstub[j / 2] * 10000 << " " << idelta[j];
    }
  }

  double deltaChisqexact =
      drinvexact * drinv_covexact + dphi0exact * dphi0_covexact + dtexact * dt_covexact + dz0exact * dz0_covexact;

  int irinvseed = tracklet->fpgarinv().value();
  int iphi0seed = tracklet->fpgaphi0().value();

  int itseed = tracklet->fpgat().value();
  int iz0seed = tracklet->fpgaz0().value();

  int irinvfit = irinvseed + ((idrinv + (1 << settings_.fitrinvbitshift())) >> settings_.fitrinvbitshift());

  int iphi0fit = iphi0seed + (idphi0 >> settings_.fitphi0bitshift());
  int itfit = itseed + (idt >> settings_.fittbitshift());

  int iz0fit = iz0seed + (idz0 >> settings_.fitz0bitshift());

  double rinvfit = rinvseed - drinv;
  double phi0fit = phi0seed - dphi0;

  double tfit = tseed - dt;
  double z0fit = z0seed - dz0;

  double rinvfitexact = rinvseedexact - drinvexact;
  double phi0fitexact = phi0seedexact - dphi0exact;

  double tfitexact = tseedexact - dtexact;
  double z0fitexact = z0seedexact - dz0exact;

  double chisqfitexact = chisqseedexact + deltaChisqexact;

  ////////////// NEW CHISQ /////////////////////
  bool NewChisqDebug = false;
  double chisqfit = 0.0;
  uint ichisqfit = 0;

  double phifactor;
  double rzfactor;
  double iphifactor;
  double irzfactor;
  int k = 0;  // column index of D matrix

  if (NewChisqDebug) {
    edm::LogVerbatim("Tracklet") << "OG chisq: \n"
                                 << "drinv/cov = " << drinv << "/" << drinv_cov << " \n"
                                 << "dphi0/cov = " << drinv << "/" << dphi0_cov << " \n"
                                 << "dt/cov = " << drinv << "/" << dt_cov << " \n"
                                 << "dz0/cov = " << drinv << "/" << dz0_cov << "\n";
    std::string myout = "D[0][k]= ";
    for (unsigned int i = 0; i < 2 * n; i++) {
      myout += std::to_string(D[0][i]);
      myout += ", ";
    }
    edm::LogVerbatim("Tracklet") << myout;
  }

  for (unsigned int i = 0; i < n; i++) {  // loop over stubs

    phifactor = rstub[k / 2] * delta[k] / sigma[k] + D[0][k] * drinv + D[1][k] * dphi0 + D[2][k] * dt + D[3][k] * dz0;
    iphifactor = kfactor[k] * rstub[k / 2] * idelta[k] * (1 << settings_.chisqphifactbits()) / sigma[k] -
                 iD[0][k] * idrinv - iD[1][k] * idphi0 - iD[2][k] * idt - iD[3][k] * idz0;

    if (NewChisqDebug) {
      edm::LogVerbatim("Tracklet") << "delta[k]/sigma = " << delta[k] / sigma[k] << "  delta[k] = " << delta[k] << "\n"
                                   << "sum = " << phifactor - delta[k] / sigma[k] << "  drinvterm = " << D[0][k] * drinv
                                   << "  dphi0term = " << D[1][k] * dphi0 << "  dtterm = " << D[2][k] * dt
                                   << "  dz0term = " << D[3][k] * dz0 << "\n  phifactor = " << phifactor;
    }

    chisqfit += phifactor * phifactor;
    ichisqfit += iphifactor * iphifactor / (1 << (2 * settings_.chisqphifactbits() - 4));

    k++;

    rzfactor = delta[k] / sigma[k] + D[0][k] * drinv + D[1][k] * dphi0 + D[2][k] * dt + D[3][k] * dz0;
    irzfactor = kfactor[k] * idelta[k] * (1 << settings_.chisqzfactbits()) / sigma[k] - iD[0][k] * idrinv -
                iD[1][k] * idphi0 - iD[2][k] * idt - iD[3][k] * idz0;

    if (NewChisqDebug) {
      edm::LogVerbatim("Tracklet") << "delta[k]/sigma = " << delta[k] / sigma[k] << "  delta[k] = " << delta[k] << "\n"
                                   << "sum = " << rzfactor - delta[k] / sigma[k] << "  drinvterm = " << D[0][k] * drinv
                                   << "  dphi0term = " << D[1][k] * dphi0 << "  dtterm = " << D[2][k] * dt
                                   << "  dz0term = " << D[3][k] * dz0 << "\n  rzfactor = " << rzfactor;
    }

    chisqfit += rzfactor * rzfactor;
    ichisqfit += irzfactor * irzfactor / (1 << (2 * settings_.chisqzfactbits() - 4));

    k++;
  }

  if (settings_.writeMonitorData("ChiSq")) {
    globals_->ofstream("chisq.txt") << asinh(itfit * settings_.ktpars()) << " " << chisqfit << " " << ichisqfit / 16.0
                                    << endl;
  }

  // Chisquare per DOF capped out at 11 bits, so 15 is an educated guess
  if (ichisqfit >= (1 << 15)) {
    if (NewChisqDebug) {
      edm::LogVerbatim("Tracklet") << "CHISQUARE (" << ichisqfit << ") LARGER THAN 11 BITS!";
    }
    ichisqfit = (1 << 15) - 1;
  }

  // Eliminate lower bits to fit in 8 bits
  ichisqfit = ichisqfit >> 7;
  // Probably redundant... enforce 8 bit cap
  if (ichisqfit >= (1 << 8))
    ichisqfit = (1 << 8) - 1;

  double phicrit = phi0fit - asin(0.5 * settings_.rcrit() * rinvfit);
  bool keep = (phicrit > settings_.phicritmin()) && (phicrit < settings_.phicritmax());

  if (!keep) {
    return;
  }

  // NOTE: setFitPars in Tracklet.h now accepts chi2 r-phi and chi2 r-z values. This class only has access
  // to the composite chi2. When setting fit parameters on a tracklet, this places all of the chi2 into the
  // r-phi fit, and sets the r-z fit value to zero.
  //
  // This is also true for the call to setFitPars in trackFitFake.
  tracklet->setFitPars(rinvfit,
                       phi0fit,
                       0.0,
                       tfit,
                       z0fit,
                       chisqfit,
                       0.0,
                       rinvfitexact,
                       phi0fitexact,
                       0.0,
                       tfitexact,
                       z0fitexact,
                       chisqfitexact,
                       0.0,
                       irinvfit,
                       iphi0fit,
                       0,
                       itfit,
                       iz0fit,
                       ichisqfit,
                       0,
                       0);
}

void FitTrack::trackFitFake(Tracklet* tracklet, std::vector<const Stub*>&, std::vector<std::pair<int, int>>&) {
  tracklet->setFitPars(tracklet->rinvapprox(),
                       tracklet->phi0approx(),
                       tracklet->d0approx(),
                       tracklet->tapprox(),
                       tracklet->z0approx(),
                       0.0,
                       0.0,
                       tracklet->rinv(),
                       tracklet->phi0(),
                       tracklet->d0(),
                       tracklet->t(),
                       tracklet->z0(),
                       0.0,
                       0.0,
                       tracklet->fpgarinv().value(),
                       tracklet->fpgaphi0().value(),
                       tracklet->fpgad0().value(),
                       tracklet->fpgat().value(),
                       tracklet->fpgaz0().value(),
                       0,
                       0,
                       0);
  return;
}

std::vector<Tracklet*> FitTrack::orderedMatches(vector<FullMatchMemory*>& fullmatch) {
  std::vector<Tracklet*> tmp;

  std::vector<unsigned int> indexArray;
  for (auto& imatch : fullmatch) {
    //check that we have correct order
    if (imatch->nMatches() > 1) {
      for (unsigned int j = 0; j < imatch->nMatches() - 1; j++) {
        assert(imatch->getTracklet(j)->TCID() <= imatch->getTracklet(j + 1)->TCID());
      }
    }

    if (settings_.debugTracklet() && imatch->nMatches() != 0) {
      edm::LogVerbatim("Tracklet") << "orderedMatches: " << imatch->getName() << " " << imatch->nMatches();
    }

    indexArray.push_back(0);
  }

  int bestIndex = -1;
  do {
    int bestTCID = -1;
    bestIndex = -1;
    for (unsigned int i = 0; i < fullmatch.size(); i++) {
      if (indexArray[i] >= fullmatch[i]->nMatches()) {
        //skip as we were at the end
        continue;
      }
      int TCID = fullmatch[i]->getTracklet(indexArray[i])->TCID();
      if (TCID < bestTCID || bestTCID < 0) {
        bestTCID = TCID;
        bestIndex = i;
      }
    }
    if (bestIndex != -1) {
      tmp.push_back(fullmatch[bestIndex]->getTracklet(indexArray[bestIndex]));
      indexArray[bestIndex]++;
    }
  } while (bestIndex != -1);

  for (unsigned int i = 0; i < tmp.size(); i++) {
    if (i > 0) {
      //This allows for equal TCIDs. This means that we can e.g. have a track seeded in L1L2 that projects to both L3 and D4.
      //The algorithm will pick up the first hit and drop the second.
      if (tmp[i - 1]->TCID() > tmp[i]->TCID()) {
        edm::LogVerbatim("Tracklet") << "Wrong TCID ordering in " << getName() << " : " << tmp[i - 1]->TCID() << " "
                                     << tmp[i]->TCID();
      }
    }
  }

  return tmp;
}

void FitTrack::execute(unsigned int iSector) {
  // merge
  const std::vector<Tracklet*>& matches1 = orderedMatches(fullmatch1_);
  const std::vector<Tracklet*>& matches2 = orderedMatches(fullmatch2_);
  const std::vector<Tracklet*>& matches3 = orderedMatches(fullmatch3_);
  const std::vector<Tracklet*>& matches4 = orderedMatches(fullmatch4_);

  iSector_ = iSector;

  if (settings_.debugTracklet() && (matches1.size() + matches2.size() + matches3.size() + matches4.size()) > 0) {
    for (auto& imatch : fullmatch1_) {
      edm::LogVerbatim("Tracklet") << imatch->getName() << " " << imatch->nMatches();
    }
    edm::LogVerbatim("Tracklet") << getName() << " matches : " << matches1.size() << " " << matches2.size() << " "
                                 << matches3.size() << " " << matches4.size();
  }

  unsigned int indexArray[4];
  for (unsigned int i = 0; i < 4; i++) {
    indexArray[i] = 0;
  }

  int countAll = 0;
  int countFit = 0;

  Tracklet* bestTracklet = nullptr;
  do {
    countAll++;
    bestTracklet = nullptr;

    if (indexArray[0] < matches1.size()) {
      if (bestTracklet == nullptr) {
        bestTracklet = matches1[indexArray[0]];
      } else {
        if (matches1[indexArray[0]]->TCID() < bestTracklet->TCID())
          bestTracklet = matches1[indexArray[0]];
      }
    }

    if (indexArray[1] < matches2.size()) {
      if (bestTracklet == nullptr) {
        bestTracklet = matches2[indexArray[1]];
      } else {
        if (matches2[indexArray[1]]->TCID() < bestTracklet->TCID())
          bestTracklet = matches2[indexArray[1]];
      }
    }

    if (indexArray[2] < matches3.size()) {
      if (bestTracklet == nullptr) {
        bestTracklet = matches3[indexArray[2]];
      } else {
        if (matches3[indexArray[2]]->TCID() < bestTracklet->TCID())
          bestTracklet = matches3[indexArray[2]];
      }
    }

    if (indexArray[3] < matches4.size()) {
      if (bestTracklet == nullptr) {
        bestTracklet = matches4[indexArray[3]];
      } else {
        if (matches4[indexArray[3]]->TCID() < bestTracklet->TCID())
          bestTracklet = matches4[indexArray[3]];
      }
    }

    if (bestTracklet == nullptr)
      break;

    //Counts total number of matched hits
    int nMatches = 0;

    //Counts unique hits in each layer
    int nMatchesUniq = 0;
    bool match = false;

    while (indexArray[0] < matches1.size() && matches1[indexArray[0]] == bestTracklet) {
      indexArray[0]++;
      nMatches++;
      match = true;
    }

    if (match)
      nMatchesUniq++;
    match = false;

    while (indexArray[1] < matches2.size() && matches2[indexArray[1]] == bestTracklet) {
      indexArray[1]++;
      nMatches++;
      match = true;
    }

    if (match)
      nMatchesUniq++;
    match = false;

    while (indexArray[2] < matches3.size() && matches3[indexArray[2]] == bestTracklet) {
      indexArray[2]++;
      nMatches++;
      match = true;
    }

    if (match)
      nMatchesUniq++;
    match = false;

    while (indexArray[3] < matches4.size() && matches4[indexArray[3]] == bestTracklet) {
      indexArray[3]++;
      nMatches++;
      match = true;
    }

    if (match)
      nMatchesUniq++;

    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " : nMatches = " << nMatches << " nMatchesUniq = " << nMatchesUniq
                                   << " " << asinh(bestTracklet->t());
    }

    std::vector<const Stub*> trackstublist;
    std::vector<std::pair<int, int>> stubidslist;
    if ((bestTracklet->getISeed() >= (int)N_SEED_PROMPT && nMatchesUniq >= 1) ||
        nMatchesUniq >= 2) {  //For seeds index >=8 (triplet seeds), there are three stubs associated from start.
      countFit++;

#ifdef USEHYBRID
      trackFitKF(bestTracklet, trackstublist, stubidslist);
#else
      if (settings_.fakefit()) {
        trackFitFake(bestTracklet, trackstublist, stubidslist);
      } else {
        trackFitChisq(bestTracklet, trackstublist, stubidslist);
      }
#endif

      if (settings_.removalType() == "merge") {
        trackfit_->addStubList(trackstublist);
        trackfit_->addStubidsList(stubidslist);
        bestTracklet->setTrackIndex(trackfit_->nTracks());
        trackfit_->addTrack(bestTracklet);
      } else if (bestTracklet->fit()) {
        assert(trackfit_ != nullptr);
        if (settings_.writeMonitorData("Seeds")) {
          ofstream fout("seeds.txt", ofstream::app);
          fout << __FILE__ << ":" << __LINE__ << " " << name_ << "_"
               << " " << bestTracklet->getISeed() << endl;
          fout.close();
        }
        bestTracklet->setTrackIndex(trackfit_->nTracks());
        trackfit_->addTrack(bestTracklet);
      }
    }

  } while (bestTracklet != nullptr);

  if (settings_.writeMonitorData("FT")) {
    globals_->ofstream("fittrack.txt") << getName() << " " << countAll << " " << countFit << endl;
  }
}

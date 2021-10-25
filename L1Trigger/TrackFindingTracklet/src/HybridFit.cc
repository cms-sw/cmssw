#include "L1Trigger/TrackFindingTracklet/interface/HybridFit.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Stub.h"
#include "L1Trigger/TrackFindingTracklet/interface/L1TStub.h"

#ifdef USEHYBRID
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace trklet;

HybridFit::HybridFit(unsigned int iSector, Settings const& settings, Globals* globals) : settings_(settings) {
  iSector_ = iSector;
  globals_ = globals;
}

void HybridFit::Fit(Tracklet* tracklet, std::vector<const Stub*>& trackstublist) {
  if (settings_.fakefit()) {
    vector<const L1TStub*> l1stubsFromFitTrack;
    for (unsigned int k = 0; k < trackstublist.size(); k++) {
      const L1TStub* L1stub = trackstublist[k]->l1tstub();
      l1stubsFromFitTrack.push_back(L1stub);
    }
    tracklet->setFitPars(tracklet->rinvapprox(),
                         tracklet->phi0approx(),
                         tracklet->d0approx(),
                         tracklet->tapprox(),
                         tracklet->z0approx(),
                         0.,
                         0.,
                         tracklet->rinv(),
                         tracklet->phi0(),
                         tracklet->d0(),
                         tracklet->t(),
                         tracklet->z0(),
                         0.,
                         0.,
                         tracklet->fpgarinv().value(),
                         tracklet->fpgaphi0().value(),
                         tracklet->fpgad0().value(),
                         tracklet->fpgat().value(),
                         tracklet->fpgaz0().value(),
                         0,
                         0,
                         0,
                         l1stubsFromFitTrack);
    return;
  }

  std::vector<tmtt::Stub*> TMTTstubs;
  std::map<unsigned int, const L1TStub*> L1StubIndices;
  unsigned int L1stubID = 0;

  if (globals_->tmttSettings() == nullptr) {
    if (settings_.printDebugKF())
      edm::LogVerbatim("L1track") << "Creating TMTT::Settings in HybridFit::Fit";
    globals_->tmttSettings() = make_unique<tmtt::Settings>();
    globals_->tmttSettings()->setMagneticField(settings_.bfield());
  }

  const tmtt::Settings& TMTTsettings = *globals_->tmttSettings();

  int kf_phi_sec = iSector_;

  for (unsigned int k = 0; k < trackstublist.size(); k++) {
    const L1TStub* L1stubptr = trackstublist[k]->l1tstub();

    double kfphi = L1stubptr->phi();
    double kfr = L1stubptr->r();
    double kfz = L1stubptr->z();
    double kfbend = L1stubptr->bend();
    bool psmodule = L1stubptr->isPSmodule();
    unsigned int iphi = L1stubptr->iphi();
    double alpha = L1stubptr->alpha(settings_.stripPitch(psmodule));
    bool isTilted = L1stubptr->isTilted();

    bool isBarrel = trackstublist[k]->layerdisk() < N_LAYER;
    int kflayer;

    if (isBarrel) {  // Barrel-specific
      kflayer = L1stubptr->layer() + 1;
      if (settings_.printDebugKF())
        edm::LogVerbatim("L1track") << "Will create layer stub with : ";
    } else {  // Disk-specific
      kflayer = abs(L1stubptr->disk());
      if (kfz > 0) {
        kflayer += 10;
      } else {
        kflayer += 20;
      }
      if (settings_.printDebugKF())
        edm::LogVerbatim("L1track") << "Will create disk stub with : ";
    }

    float stripPitch = settings_.stripPitch(psmodule);
    float stripLength = settings_.stripLength(psmodule);
    unsigned int nStrips = settings_.nStrips(psmodule);

    if (settings_.printDebugKF()) {
      edm::LogVerbatim("L1track") << kfphi << " " << kfr << " " << kfz << " " << kfbend << " " << kflayer << " "
                                  << isBarrel << " " << psmodule << " " << isTilted << " \n"
                                  << stripPitch << " " << stripLength << " " << nStrips;
    }

    unsigned int uniqueStubIndex = 1000 * L1stubID + L1stubptr->allStubIndex();
    tmtt::Stub* TMTTstubptr = new tmtt::Stub(&TMTTsettings,
                                             uniqueStubIndex,
                                             kfphi,
                                             kfr,
                                             kfz,
                                             kfbend,
                                             iphi,
                                             -alpha,
                                             kflayer,
                                             kf_phi_sec,
                                             psmodule,
                                             isBarrel,
                                             isTilted,
                                             stripPitch,
                                             stripLength,
                                             nStrips);
    TMTTstubs.push_back(TMTTstubptr);
    L1StubIndices[uniqueStubIndex] = L1stubptr;
    L1stubID++;
  }

  if (settings_.printDebugKF()) {
    edm::LogVerbatim("L1track") << "Made TMTTstubs: trackstublist.size() = " << trackstublist.size();
  }

  double kfrinv = tracklet->rinvapprox();
  double kfphi0 = tracklet->phi0approx();
  double kfz0 = tracklet->z0approx();
  double kft = tracklet->tapprox();
  double kfd0 = tracklet->d0approx();

  if (settings_.printDebugKF()) {
    edm::LogVerbatim("L1track") << "tracklet phi0 = " << kfphi0 << "\n"
                                << "iSector = " << iSector_ << "\n"
                                << "dphisectorHG = " << settings_.dphisectorHG();
  }

  // KF wants global phi0, not phi0 measured with respect to lower edge of sector (Tracklet convention).
  kfphi0 = reco::reduceRange(kfphi0 + iSector_ * settings_.dphisector() - 0.5 * settings_.dphisectorHG());

  std::pair<float, float> helixrphi(kfrinv / (0.01 * settings_.c() * settings_.bfield()), kfphi0);
  std::pair<float, float> helixrz(kfz0, kft);

  // KF HLS uses HT mbin (which is binned q/Pt) to allow for scattering. So estimate it from tracklet.
  double chargeOverPt = helixrphi.first;
  int mBin = std::floor(TMTTsettings.houghNbinsPt() / 2) +
             std::floor((TMTTsettings.houghNbinsPt() / 2) * chargeOverPt / (1. / TMTTsettings.houghMinPt()));
  mBin = max(min(mBin, int(TMTTsettings.houghNbinsPt() - 1)), 0);  // protect precision issues.
  std::pair<unsigned int, unsigned int> celllocation(mBin, 1);

  // Get range in z of tracks covered by this sector at chosen radius from beam-line
  const vector<double> etaRegions = TMTTsettings.etaRegions();
  const float chosenRofZ = TMTTsettings.chosenRofZ();

  float kfzRef = kfz0 + chosenRofZ * kft;

  unsigned int kf_eta_reg = 0;
  for (unsigned int iEtaSec = 1; iEtaSec < etaRegions.size() - 1; iEtaSec++) {  // Doesn't apply eta < 2.4 cut.
    const float etaMax = etaRegions[iEtaSec];
    const float zRefMax = chosenRofZ / tan(2. * atan(exp(-etaMax)));
    if (kfzRef > zRefMax)
      kf_eta_reg = iEtaSec;
  }

  tmtt::L1track3D l1track3d(
      &TMTTsettings, TMTTstubs, celllocation, helixrphi, helixrz, kfd0, kf_phi_sec, kf_eta_reg, 1, false);
  unsigned int seedType = tracklet->getISeed();
  unsigned int numPS = tracklet->PSseed();  // Function PSseed() is out of date!
  l1track3d.setSeedLayerType(seedType);
  l1track3d.setSeedPS(numPS);

  if (globals_->tmttKFParamsComb() == nullptr) {
    if (settings_.printDebugKF())
      edm::LogVerbatim("L1track") << "Will make KFParamsComb for " << settings_.nHelixPar() << " param fit";
    globals_->tmttKFParamsComb() = make_unique<tmtt::KFParamsComb>(&TMTTsettings, settings_.nHelixPar(), "KFfitter");
  }

  tmtt::KFParamsComb& fitterKF = *globals_->tmttKFParamsComb();

  // Call Kalman fit
  tmtt::L1fittedTrack fittedTrk = fitterKF.fit(l1track3d);

  if (fittedTrk.accepted()) {
    tmtt::KFTrackletTrack trk = fittedTrk.returnKFTrackletTrack();

    if (settings_.printDebugKF())
      edm::LogVerbatim("L1track") << "Done with Kalman fit. Pars: pt = " << trk.pt()
                                  << ", 1/2R = " << settings_.bfield() * 3 * trk.qOverPt() / 2000
                                  << ", phi0 = " << trk.phi0() << ", eta = " << trk.eta() << ", z0 = " << trk.z0()
                                  << ", chi2 = " << trk.chi2() << ", accepted = " << trk.accepted();

    double d0, chi2rphi, phi0, qoverpt = -999;
    if (trk.done_bcon()) {
      d0 = trk.d0_bcon();
      chi2rphi = trk.chi2rphi_bcon();
      phi0 = trk.phi0_bcon();
      qoverpt = trk.qOverPt_bcon();
    } else {
      d0 = trk.d0();
      chi2rphi = trk.chi2rphi();
      phi0 = trk.phi0();
      qoverpt = trk.qOverPt();
    }

    // Tracklet wants phi0 with respect to lower edge of sector, not global phi0.
    double phi0fit = reco::reduceRange(phi0 - iSector_ * 2 * M_PI / N_SECTOR + 0.5 * settings_.dphisectorHG());
    double rinvfit = 0.01 * settings_.c() * settings_.bfield() * qoverpt;

    int irinvfit = rinvfit / settings_.krinvpars();
    int iphi0fit = phi0fit / settings_.kphi0pars();
    int itanlfit = trk.tanLambda() / settings_.ktpars();
    int iz0fit = trk.z0() / settings_.kz0pars();
    int id0fit = d0 / settings_.kd0pars();
    int ichi2rphifit = chi2rphi / 16;
    int ichi2rzfit = trk.chi2rz() / 16;

    const vector<const tmtt::Stub*>& stubsFromFit = trk.stubs();
    vector<const L1TStub*> l1stubsFromFit;
    for (const tmtt::Stub* s : stubsFromFit) {
      unsigned int IDf = s->index();
      const L1TStub* l1s = L1StubIndices.at(IDf);
      l1stubsFromFit.push_back(l1s);
    }

    tracklet->setFitPars(rinvfit,
                         phi0fit,
                         d0,
                         trk.tanLambda(),
                         trk.z0(),
                         chi2rphi,
                         trk.chi2rz(),
                         rinvfit,
                         phi0fit,
                         d0,
                         trk.tanLambda(),
                         trk.z0(),
                         chi2rphi,
                         trk.chi2rz(),
                         irinvfit,
                         iphi0fit,
                         id0fit,
                         itanlfit,
                         iz0fit,
                         ichi2rphifit,
                         ichi2rzfit,
                         trk.hitPattern(),
                         l1stubsFromFit);
  } else {
    if (settings_.printDebugKF()) {
      edm::LogVerbatim("L1track") << "FitTrack:KF rejected track";
    }
  }

  for (const tmtt::Stub* s : TMTTstubs) {
    delete s;
  }
}
#endif

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "L1Trigger/TrackFindingTMTT/interface/TP.h"
#include "L1Trigger/TrackFindingTMTT/interface/Stub.h"
#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "L1Trigger/TrackFindingTMTT/interface/Utility.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <array>

using namespace std;

namespace tmtt {

  //=== Store useful info about this tracking particle

  TP::TP(const TrackingParticlePtr& tpPtr, unsigned int index_in_vTPs, const Settings* settings)
      : trackingParticlePtr_(tpPtr),
        index_in_vTPs_(index_in_vTPs),
        settings_(settings),
        pdgId_(tpPtr->pdgId()),
        charge_(tpPtr->charge()),
        mass_(tpPtr->mass()),
        pt_(tpPtr->pt()),
        eta_(tpPtr->eta()),
        theta_(tpPtr->theta()),
        tanLambda_(1. / tan(theta_)),
        phi0_(tpPtr->phi()),
        vx_(tpPtr->vertex().x()),
        vy_(tpPtr->vertex().y()),
        vz_(tpPtr->vertex().z()),
        d0_(vx_ * sin(phi0_) - vy_ * cos(phi0_)),                      // Copied from CMSSW class TrackBase::d0().
        z0_(vz_ - (vx_ * cos(phi0_) + vy_ * sin(phi0_)) * tanLambda_)  // Copied from CMSSW class TrackBase::dz().
  {
    const vector<SimTrack>& vst = tpPtr->g4Tracks();
    EncodedEventId eid = vst.at(0).eventId();
    inTimeBx_ = (eid.bunchCrossing() == 0);  // TP from in-time or out-of-time Bx.
    physicsCollision_ = (eid.event() == 0);  // TP from physics collision or from pileup.

    this->fillUse();        // Fill use_ flag, indicating if TP is worth keeping.
    this->fillUseForEff();  // Fill useForEff_ flag, indicating if TP is good for tracking efficiency measurement.
  }

  //=== Fill truth info with association from tracking particle to stubs.

  void TP::fillTruth(const list<Stub>& vStubs) {
    for (const Stub& s : vStubs) {
      for (const TP* tp_i : s.assocTPs()) {
        if (tp_i->index() == this->index())
          assocStubs_.push_back(&s);
      }
    }

    this->fillUseForAlgEff();  // Fill useForAlgEff_ flag.

    this->calcNumLayers();  // Calculate number of tracker layers this TP has stubs in.
  }

  //=== Check if this tracking particle is worth keeping.
  //=== (i.e. If there is the slightest chance of reconstructing it, so as to measure fake rate).

  void TP::fillUse() {
    constexpr bool useOnlyInTimeParticles = false;
    constexpr bool useOnlyTPfromPhysicsCollisionFalse = false;
    // Use looser cuts here those those used for tracking efficiency measurement.
    // Keep only those TP that have a chance (allowing for finite track resolution) of being reconstructed as L1 tracks. L1 tracks not matching these TP will be defined as fake.

    // Include all possible particle types here, as if some are left out, L1 tracks matching one of missing types will be declared fake.
    constexpr std::array<int, 5> genPdgIdsAllUnsigned = {{11, 13, 211, 321, 2212}};
    vector<int> genPdgIdsAll;
    for (const int& iPdg : genPdgIdsAllUnsigned) {
      genPdgIdsAll.push_back(iPdg);
      genPdgIdsAll.push_back(-iPdg);
    }

    // Range big enough to include all TP needed to measure tracking efficiency
    // and big enough to include any TP that might be reconstructed for fake rate measurement.
    constexpr float ptMinScale = 0.7;
    const float ptMin = min(settings_->genMinPt(), ptMinScale * settings_->houghMinPt());
    constexpr double ptMax = 9.9e9;
    const float etaExtra = 0.2;
    const float etaMax = max(settings_->genMaxAbsEta(), etaExtra + std::abs(settings_->etaRegions()[0]));
    constexpr double fixedVertRcut = 10.;
    constexpr double fixedVertZcut = 35.;

    static const TrackingParticleSelector trackingParticleSelector(ptMin,
                                                                   ptMax,
                                                                   -etaMax,
                                                                   etaMax,
                                                                   max(fixedVertRcut, settings_->genMaxVertR()),
                                                                   max(fixedVertZcut, settings_->genMaxVertZ()),
                                                                   0,
                                                                   useOnlyTPfromPhysicsCollisionFalse,
                                                                   useOnlyInTimeParticles,
                                                                   true,
                                                                   false,
                                                                   genPdgIdsAll);

    use_ = trackingParticleSelector(*trackingParticlePtr_);
  }

  //=== Check if this tracking particle can be used to measure the L1 tracking efficiency.

  void TP::fillUseForEff() {
    useForEff_ = false;
    if (use_) {
      constexpr bool useOnlyInTimeParticles = true;
      constexpr bool useOnlyTPfromPhysicsCollision = true;
      constexpr double ptMax = 9.9e9;
      static const TrackingParticleSelector trackingParticleSelector(settings_->genMinPt(),
                                                                     ptMax,
                                                                     -settings_->genMaxAbsEta(),
                                                                     settings_->genMaxAbsEta(),
                                                                     settings_->genMaxVertR(),
                                                                     settings_->genMaxVertZ(),
                                                                     0,
                                                                     useOnlyTPfromPhysicsCollision,
                                                                     useOnlyInTimeParticles,
                                                                     true,
                                                                     false,
                                                                     settings_->genPdgIds());

      useForEff_ = trackingParticleSelector(*trackingParticlePtr_);

      // Add additional cut on particle transverse impact parameter.
      if (std::abs(d0_) > settings_->genMaxD0())
        useForEff_ = false;
      if (std::abs(z0_) > settings_->genMaxZ0())
        useForEff_ = false;
    }
  }

  //=== Check if this tracking particle can be used to measure the L1 tracking algorithmic efficiency (makes stubs in enough layers).

  void TP::fillUseForAlgEff() {
    useForAlgEff_ = false;
    if (useForEff_) {
      useForAlgEff_ = (Utility::countLayers(settings_, assocStubs_, true) >= settings_->genMinStubLayers());
    }
  }

  //== Estimated phi angle at which TP trajectory crosses the module containing the stub.

  float TP::trkPhiAtStub(const Stub* stub) const {
    float trkPhi = phi0_ - this->dphi(this->trkRAtStub(stub));
    return trkPhi;
  }

  //== Estimated r coord. at which TP trajectory crosses the module containing the given stub.
  //== Only works for modules orientated parallel or perpendicular to beam-axis,
  //== and makes the approximation that tracks are straight-lines in r-z plane.

  float TP::trkRAtStub(const Stub* stub) const {
    float rTrk = (stub->barrel()) ? stub->r() : (stub->z() - z0_) / tanLambda_;
    return rTrk;
  }

  //== Estimated z coord. at which TP trajectory crosses the module containing the given stub.
  //== Only works for modules orientated parallel or perpendicular to beam-axis,
  //== and makes the approximation that tracks are straight-lines in r-z plane.

  float TP::trkZAtStub(const Stub* stub) const {
    float zTrk = (stub->barrel()) ? z0_ + tanLambda_ * stub->r() : stub->z();
    return zTrk;
  }

  void TP::fillNearestJetInfo(const reco::GenJetCollection* genJets) {
    double minDR = 999.;
    double ptOfNearestJet = -1;

    reco::GenJetCollection::const_iterator iterGenJet;
    for (iterGenJet = genJets->begin(); iterGenJet != genJets->end(); ++iterGenJet) {
      reco::GenJet myJet = reco::GenJet(*iterGenJet);

      // Don't consider GenJets failing these cuts.
      constexpr float minPt = 30.0;
      constexpr float maxEta = 2.5;

      if (myJet.pt() > minPt && std::abs(myJet.eta()) > maxEta) {
        double deltaR = reco::deltaR(this->eta(), this->phi0(), myJet.eta(), myJet.phi());

        if (deltaR < minDR) {
          minDR = deltaR;
          ptOfNearestJet = myJet.pt();
        }
      }
    }

    // Only consider GenJets within this distance of TP.
    constexpr float cutDR = 0.4;
    tpInJet_ = (minDR < cutDR);
    nearestJetPt_ = tpInJet_ ? ptOfNearestJet : -1.;
  }

}  // namespace tmtt

#include "RecoTauTag/RecoTau/interface/RecoTauIsolationMasking.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"

namespace reco::tau {

namespace {
class DRSorter {
  public:
    DRSorter(const reco::Candidate::LorentzVector& axis):axis_(axis){}
    template<typename T>
    bool operator()(const T& t1, const T& t2) const {
      return reco::deltaR(t1->p4(), axis_) < reco::deltaR(t2->p4(), axis_);
    }
  private:
    const reco::Candidate::LorentzVector axis_;
};
// Sort by descending pt
class PtSorter {
  public:
    PtSorter(){}
    template<typename T>
    bool operator()(const T& t1, const T& t2) const {
      return t1->pt() > t2->pt();
    }
};

// Check if an object is within DR of a track collection
class MultiTrackDRFilter {
  public:
  MultiTrackDRFilter(double deltaR, const std::vector<reco::PFCandidatePtr>& trks)
      :deltaR_(deltaR),tracks_(trks){}
    template <typename T>
    bool operator()(const T& t) const {
      for(auto const& trk : tracks_) {
        if (reco::deltaR(trk->p4(), t->p4()) < deltaR_)
          return true;
      }
      return false;
    }
  private:
    double deltaR_;
  const std::vector<reco::PFCandidatePtr>& tracks_;
};

double square(double x) { return x*x; }

template<typename T>
std::vector<reco::PFCandidatePtr> convertRefCollection(const T& coll) {
  std::vector<reco::PFCandidatePtr> output;
  output.reserve(coll.size());
  for(auto const& cand : coll) {
    output.push_back(cand);
  }
  return output;
}
}

RecoTauIsolationMasking::RecoTauIsolationMasking(
    const edm::ParameterSet& pset):resolutions_(new PFEnergyResolution) {
  ecalCone_ = pset.getParameter<double>("ecalCone");
  hcalCone_ = pset.getParameter<double>("hcalCone");
  finalHcalCone_ = pset.getParameter<double>("finalHcalCone");
  maxSigmas_ = pset.getParameter<double>("maxSigmas");
}

// Need to explicitly define this in the .cc so we can use auto_ptr + forward
RecoTauIsolationMasking::~RecoTauIsolationMasking(){}

RecoTauIsolationMasking::IsoMaskResult
RecoTauIsolationMasking::mask(const reco::PFTau& tau) const {
  IsoMaskResult output;

  typedef std::list<reco::PFCandidatePtr> PFCandList;
  // Copy original iso collections.
  std::copy(tau.isolationPFGammaCands().begin(),
      tau.isolationPFGammaCands().end(), std::back_inserter(output.gammas));
  std::copy(tau.isolationPFNeutrHadrCands().begin(),
      tau.isolationPFNeutrHadrCands().end(),
      std::back_inserter(output.h0s));

  std::vector<PFCandList*> courses;
  courses.push_back(&(output.h0s));
  courses.push_back(&(output.gammas));
  // Mask using each one of the tracks
  for(auto const& track : tau.signalPFChargedHadrCands()) {
    double trackerEnergy = track->energy();
    double linkedEcalEnergy = track->ecalEnergy();
    double linkedHcalEnergy = track->hcalEnergy();
    math::XYZPointF posAtCalo = track->positionAtECALEntrance();
    // Get the linked calo energies & their errors
    double linkedSumErrSquared = 0;
    linkedSumErrSquared += square(
        resolutions_->getEnergyResolutionEm(linkedEcalEnergy, posAtCalo.eta()));
    linkedSumErrSquared += square(
        resolutions_->getEnergyResolutionHad(
          linkedHcalEnergy, posAtCalo.eta(), posAtCalo.phi()));

    // energyDelta is the difference between associated Calo and tracker energy
    double energyDelta = linkedEcalEnergy + linkedHcalEnergy - trackerEnergy;

    // Sort the neutral hadrons by DR to the track
    //DRSorter sorter(track->p4());
    PtSorter sorter;

    for(auto* course : courses) {
      // Sort by deltaR to the current track
      course->sort(sorter);
      PFCandList::iterator toEatIter = course->begin();
      // While there are still candidates to eat in this course and they are
      // within the cone.
      while (toEatIter != course->end()) {
        const reco::PFCandidate& toEat = **toEatIter;
        double toEatEnergy = toEat.energy();
        double toEatErrorSq = square(resolution(toEat));
        // Check if we can absorb this candidate into the track.
        if (inCone(*track, **toEatIter) &&
            (energyDelta + toEatEnergy)/std::sqrt(
              linkedSumErrSquared + toEatErrorSq) < maxSigmas_ ) {
          energyDelta += toEatEnergy;
          linkedSumErrSquared += toEatErrorSq;
          toEatIter = course->erase(toEatIter);
        } else {
          // otherwise skip to the next one
          ++toEatIter;
        }
      }
    }
  }
  // Filter out any final HCAL objects with in cones about the tracks.
  // This removes upward fluctuating HCAL objects
  if (finalHcalCone_ > 0) {
    MultiTrackDRFilter hcalFinalFilter(finalHcalCone_,
        tau.signalPFChargedHadrCands());
    std::remove_if(output.h0s.begin(), output.h0s.end(), hcalFinalFilter);
  }
  return output;
}

double RecoTauIsolationMasking::resolution(
    const reco::PFCandidate& cand) const {
  if (cand.particleId() == reco::PFCandidate::h0) {
    // NB for HCAL it returns relative energy
    return cand.energy()*resolutions_->getEnergyResolutionHad(cand.energy(),
        cand.eta(), cand.phi());
  } else if (cand.particleId() == reco::PFCandidate::gamma) {
    return resolutions_->getEnergyResolutionEm(cand.energy(), cand.eta());
  } else if (cand.particleId() == reco::PFCandidate::e) {
    // FIXME what is the electron resolution??
    return 0.15;
  } else {
    edm::LogWarning("IsoMask::res (bad pf id)")
      << "Unknown PF ID: " << cand.particleId();
  }
  return -1;
}

bool RecoTauIsolationMasking::inCone(const reco::PFCandidate& track,
    const reco::PFCandidate& cand) const {
  double openingDR = reco::deltaR(track.positionAtECALEntrance(), cand.p4());
  if (cand.particleId() == reco::PFCandidate::h0) {
    return (openingDR < hcalCone_);
  } else if (cand.particleId() == reco::PFCandidate::gamma ||
      cand.particleId() == reco::PFCandidate::e) {
    return openingDR < ecalCone_;
  } else {
    edm::LogWarning("IsoMask::inCone (bad pf id)")
      << "Unknown PF ID: " << cand.particleId()
      << " " <<  reco::PFCandidate::e;
  }
  return -1;
}

} // end namespace reco::tau

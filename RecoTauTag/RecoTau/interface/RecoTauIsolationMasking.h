#ifndef RecoTauTag_RecoTau_RecoTauIsolationMasking_h
#define RecoTauTag_RecoTau_RecoTauIsolationMasking_h

/*
 * Mask isolation quantities in PFTau objects
 *
 * The masking is one by collecting HCAL and ECAL PF objects in cones about the
 * charged hadrons associated to the tau.  The HCAL and ECAL objects are then
 * eaten, (HCAL first, ordered by DR from track), until the total associated
 * calo energy is within maxSigmas standard deviations of the track energy.
 *
 * Authors: Evan K. Friis, Christian Veelken (UC Davis)
 *
 */

#include "DataFormats/TauReco/interface/PFTau.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// foward declaration
class PFEnergyResolution;

namespace reco { namespace tau {

class RecoTauIsolationMasking {
  public:
    // Structure containing new, maksed isolation collections
    struct IsoMaskResult {
      std::list<reco::PFCandidatePtr> gammas;
      std::list<reco::PFCandidatePtr> h0s;
    };
    RecoTauIsolationMasking(const edm::ParameterSet& pset);
    ~RecoTauIsolationMasking();
    /// Return a new isolation collections with masking applied
    IsoMaskResult mask(const reco::PFTau&) const;

    void setMaxSigmas(double maxSigmas) {maxSigmas_ = maxSigmas;}
  private:
    // Get the energy resoltuion of a gamma or h0 candidate
    double resolution(const reco::PFCandidate& cand) const;
    // Check if the candidate is in the correct cone
    bool inCone(const reco::PFCandidate& track,
        const reco::PFCandidate& cand) const;

    double ecalCone_;
    double hcalCone_;
    double maxSigmas_;
    double finalHcalCone_;
    std::unique_ptr<PFEnergyResolution> resolutions_;
};

}}
#endif

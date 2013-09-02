#ifndef RecoTauTag_RecoTau_RecoTauIsolationDiscriminantPluginBase_h
#define RecoTauTag_RecoTau_RecoTauIsolationDiscriminantPluginBase_h
/*
 * RecoTauBinnedIsolation plugin
 *
 * Abstract base RecoTauDiscriminantPlugin class that computes the bin content
 * of the isolation object p_T spectra.
 *
 * The binning is parameterized by the number of pileup vertices in the event.
 *
 * The extraction of the different objects to compute the spectra of is defined
 * in the derived classes by overriding the pure abstract extractIsoObjects
 * method.
 *
 * Author: Evan K. Friis, Christian Veelken (UC Davis)
 *
 */

#include "RecoTauTag/RecoTau/interface/RecoTauDiscriminantPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco { namespace tau {

class RecoTauDiscriminationBinnedIsolation : public RecoTauDiscriminantPlugin {
  public:
    RecoTauDiscriminationBinnedIsolation(const edm::ParameterSet& pset);
    virtual ~RecoTauDiscriminationBinnedIsolation() {}
    void beginEvent();
    std::vector<double> operator()(const reco::PFTauRef& tau) const;
    // Pure abstract function to extract objects to isolate with
    virtual std::vector<reco::PFCandidatePtr> extractIsoObjects(
        const reco::PFTauRef& tau) const = 0;

  private:
    // Map of number of vertices to binning
    std::map<int, std::vector<double> > binning_;
    std::vector<double> defaultBinning_;
    // Where to get PU vertices
    size_t nVertices_;
    edm::InputTag puVtxSrc_;
};

}} // end namespace reco::tau
#endif

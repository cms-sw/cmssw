#include "RecoTauTag/RecoTau/interface/RecoTauBinnedIsolationPlugin.h"
#include <boost/foreach.hpp>
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace reco { namespace tau {

namespace {
template<typename C>
bool is_sorted(const C& collection) {
  for(size_t i = 0; i < collection.size()-1; i++) {
    if (collection[i] > collection[i+1])
      return false;
  }
  return true;
}
}

RecoTauDiscriminationBinnedIsolation::RecoTauDiscriminationBinnedIsolation(
    const edm::ParameterSet& pset):RecoTauDiscriminantPlugin(pset) {
  puVtxSrc_ = pset.getParameter<edm::InputTag>("vtxSource");
  // Configure the binning
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet binning = pset.getParameter<VPSet>("binning");
  BOOST_FOREACH(const edm::ParameterSet& bincfg, binning) {
    std::vector<double> bins = bincfg.getParameter<std::vector<double> >(
        "binLowEdges");
    int nVtx = bincfg.getParameter<int>("nPUVtx");
    // Sanity checks
    // No double entries
    if (binning_.count(nVtx)) {
      throw cms::Exception("BadIsoBinVtxConfig") << "Multiple configuraions for"
        << " vertex multiplicity: " << nVtx << " have been entered!";
    }
    // Bins are sorted
    if (!is_sorted(bins)) {
      throw cms::Exception("BadIsoBinConfig") << "The binning for vertex: "
        << nVtx << " is not in ascending order!";
    }
    binning_[nVtx] = bins;
  }
  defaultBinning_ = pset.getParameter<std::vector<double> >("defaultBinning");
}

// Load the vertices at the beginning of each event
void RecoTauDiscriminationBinnedIsolation::beginEvent() {
  edm::Handle<reco::VertexCollection> vertices_;
  evt()->getByLabel(puVtxSrc_, vertices_);
  nVertices_ = vertices_->size();
}

// Compute the result of the function
std::vector<double> RecoTauDiscriminationBinnedIsolation::operator()(
    const reco::PFTauRef& tau) const {
  // Get the binning for this event
  std::map<int, std::vector<double> >::const_iterator binningIter =
    binning_.find(nVertices_);

  const std::vector<double>* bins = NULL;
  if (binningIter != binning_.end()) {
    bins = &(binningIter->second);
  } else {
    bins = &defaultBinning_;
  }

  if (!bins) {
    throw cms::Exception("NullBinning")
      << "The binning for nVtx: " << nVertices_ << " is null!";
  }

  // Create our output spectrum
  std::vector<double> output(bins->size(), 0.0);
  // Get the desired isolation objects
  std::vector<reco::PFCandidatePtr> isoObjects = extractIsoObjects(tau);
  // Loop over each and histogram their pt
  BOOST_FOREACH(const reco::PFCandidatePtr& cand, isoObjects) {
    int highestBinLessThan = -1;
    for (size_t ibin = 0; ibin < bins->size(); ++ibin) {
      if (cand->pt() > bins->at(ibin)) {
        highestBinLessThan = ibin;
      }
    }
    if (highestBinLessThan >= 0)
      output[highestBinLessThan] += 1;
  }
  return output;
}


}} // end namespace reco::tau

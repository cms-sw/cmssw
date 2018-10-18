/*
 * ===========================================================================
 *
 *       Filename:  RecoTauPhotonFilter
 *
 *    Description:  Modify taus to recursively filter lowpt photons
 *
 *         Author:  Evan K. Friis (UC Davis)
 *
 * ===========================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

namespace reco::tau {

// Filter photons
class RecoTauPhotonFilter : public RecoTauModifierPlugin {
  public:
  explicit RecoTauPhotonFilter(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC);
    ~RecoTauPhotonFilter() override {}
    void operator()(PFTau&) const override;
  private:
    bool filter(const RecoTauPiZero* piZero,
                const reco::Candidate::LorentzVector& tau) const;
    double minPtFractionSinglePhotons_;
    double minPtFractionPiZeroes_;
};

RecoTauPhotonFilter::RecoTauPhotonFilter(
  const edm::ParameterSet& pset, edm::ConsumesCollector &&iC):RecoTauModifierPlugin(pset,std::move(iC)) {
  minPtFractionSinglePhotons_ =
      pset.getParameter<double>("minPtFractionSinglePhotons");
  minPtFractionPiZeroes_ =
      pset.getParameter<double>("minPtFractionPiZeroes");
}

// Sort container of PiZero pointers by ascending Pt
namespace {
struct PiZeroPtSorter {
  bool operator()(const RecoTauPiZero* A, const RecoTauPiZero* B) {
    return (A->pt() < B->pt());
  }
};
}

// Decide if we should filter a pi zero or not.
bool RecoTauPhotonFilter::filter( const RecoTauPiZero* piZero,
    const reco::Candidate::LorentzVector& total) const {
  if (piZero->numberOfDaughters() > 1)
    return piZero->pt()/total.pt() < minPtFractionPiZeroes_;
  return piZero->pt()/total.pt() < minPtFractionSinglePhotons_;
}

void RecoTauPhotonFilter::operator()(PFTau& tau) const {
  std::vector<const RecoTauPiZero*> signalPiZeros;
  for(auto const& piZero : tau.signalPiZeroCandidates()) {
    signalPiZeros.push_back(&piZero);
  }
  std::sort(signalPiZeros.begin(), signalPiZeros.end(), PiZeroPtSorter());
  std::vector<const RecoTauPiZero*>::const_iterator wimp =
      signalPiZeros.begin();

  // Loop until we have a sturdy enough pizero
  reco::Candidate::LorentzVector totalP4 = tau.p4();
  while(wimp != signalPiZeros.end() && filter(*wimp, totalP4)) {
    totalP4 -= (*wimp)->p4();
    ++wimp;
  }

  if (wimp != signalPiZeros.begin()) {
    // We filtered stuff, update our tau
    double ptDifference = (totalP4 - tau.p4()).pt();
    tau.setisolationPFGammaCandsEtSum(
        tau.isolationPFGammaCandsEtSum() + ptDifference);

    // Update four vector
    tau.setP4(totalP4);

    std::vector<RecoTauPiZero> toMove;
    std::vector<RecoTauPiZero> newSignal;
    std::vector<RecoTauPiZero> newIsolation;

    // Build our new objects
    for (std::vector<const RecoTauPiZero*>::const_iterator iter =
         signalPiZeros.begin(); iter != wimp; ++iter) {
      toMove.push_back(**iter);
    }
    for (std::vector<const RecoTauPiZero*>::const_iterator iter =
         wimp; iter != signalPiZeros.end(); ++iter) {
      newSignal.push_back(**iter);
    }
    // Build our new isolation collection
    std::copy(toMove.begin(), toMove.end(), std::back_inserter(newIsolation));
    std::copy(tau.isolationPiZeroCandidates().begin(),
              tau.isolationPiZeroCandidates().end(),
              std::back_inserter(newIsolation));

    // Set the collections in the taus.
    tau.setsignalPiZeroCandidates(newSignal);
    tau.setisolationPiZeroCandidates(newIsolation);

    // Now we need to deal with the gamma candidates underlying moved pizeros.
    std::vector<PFCandidatePtr> pfcandsToMove = flattenPiZeros(toMove);

    // Copy the keys to move
    std::set<size_t> keysToMove;
    for(auto const& ptr : pfcandsToMove) {
      keysToMove.insert(ptr.key());
    }

    std::vector<PFCandidatePtr> newSignalPFGammas;
    std::vector<PFCandidatePtr> newSignalPFCands;
    std::vector<PFCandidatePtr> newIsolationPFGammas = tau.isolationPFGammaCands();
    std::vector<PFCandidatePtr> newIsolationPFCands = tau.isolationPFCands();

    // Move the necessary signal pizeros - what a mess!
    for(auto const& ptr : tau.signalPFCands()) {
      if (keysToMove.count(ptr.key()))
        newIsolationPFCands.push_back(ptr);
      else
        newSignalPFCands.push_back(ptr);
    }

    for(auto const& ptr : tau.signalPFGammaCands()) {
      if (keysToMove.count(ptr.key()))
        newIsolationPFGammas.push_back(ptr);
      else
        newSignalPFGammas.push_back(ptr);
    }

    tau.setsignalPFCands(newSignalPFCands);
    tau.setsignalPFCands(newSignalPFGammas);
    tau.setisolationPFGammaCands(newIsolationPFGammas);
    tau.setisolationPFCands(newIsolationPFCands);
  }
}
}  // end namespace reco::tau
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauPhotonFilter,
    "RecoTauPhotonFilter");

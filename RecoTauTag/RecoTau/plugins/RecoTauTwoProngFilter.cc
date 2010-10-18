/*
 * ===========================================================================
 *
 *       Filename:  RecoTauTwoProngFilter
 *
 *    Description:  Modify taus remove low-pt second prongs
 *
 *         Author:  Evan K. Friis (UC Davis)
 *
 * ===========================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

namespace reco { namespace tau {

namespace {
// Delete an element from a ref vector
void deleteFrom(const PFCandidateRef ref, PFCandidateRefVector* collection) {
  PFCandidateRefVector::iterator todelete = collection->end();
  for (PFCandidateRefVector::iterator cand = collection->begin();
       cand != collection->end(); ++cand) {
    if (*cand == ref) {
      todelete = cand;
      break;
    }
  }
  if (todelete != collection->end())
    collection->erase(todelete);
}
}

class RecoTauTwoProngFilter : public RecoTauModifierPlugin {
  public:
    explicit RecoTauTwoProngFilter(const edm::ParameterSet& pset);
    virtual ~RecoTauTwoProngFilter() {}
    void operator()(PFTau&) const;
  private:
    double minPtFractionForSecondProng_;
};

RecoTauTwoProngFilter::RecoTauTwoProngFilter(const edm::ParameterSet& pset):RecoTauModifierPlugin(pset) {
  minPtFractionForSecondProng_ = pset.getParameter<double>("minPtFractionForSecondProng");
}

void RecoTauTwoProngFilter::operator()(PFTau& tau) const {
  if (tau.signalPFChargedHadrCands().size() == 2) {
    const PFCandidateRefVector &signalCharged = tau.signalPFChargedHadrCands();
    size_t indexOfHighestPt =
        (signalCharged[0]->pt() > signalCharged[1]->pt()) ? 0 : 1;
    size_t indexOfLowerPt   = ( indexOfHighestPt ) ? 0 : 1;
    double ratio = signalCharged[indexOfLowerPt]->pt()/
        signalCharged[indexOfHighestPt]->pt();

    if (ratio < minPtFractionForSecondProng_) {
      PFCandidateRef keep = signalCharged[indexOfHighestPt];
      PFCandidateRef filter = signalCharged[indexOfLowerPt];
      // Make our new signal charged candidate collection
      PFCandidateRefVector newSignalCharged;
      newSignalCharged.push_back(keep);
      PFCandidateRefVector newSignal = tau.signalPFCands();
      deleteFrom(filter, &newSignal);

      // Copy our filtered cand to isolation
      PFCandidateRefVector newIsolationCharged =
          tau.isolationPFChargedHadrCands();
      newIsolationCharged.push_back(filter);
      PFCandidateRefVector newIsolation = tau.isolationPFCands();
      newIsolation.push_back(filter);

      // Update tau members
      tau.setP4(tau.p4() - filter->p4());
      tau.setisolationPFChargedHadrCandsPtSum(
          tau.isolationPFChargedHadrCandsPtSum() - filter->pt());
      tau.setCharge(tau.charge() - filter->charge());
      // Update tau constituents
      tau.setsignalPFChargedHadrCands(newSignalCharged);
      tau.setsignalPFCands(newSignal);
      tau.setisolationPFChargedHadrCands(newIsolationCharged);
      tau.setisolationPFCands(newIsolation);
    }
  }
}
}}  // end namespace reco::tau
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
    reco::tau::RecoTauTwoProngFilter,
    "RecoTauTwoProngFilter");

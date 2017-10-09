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

namespace 
{
  // Delete an element from a ptr vector
  std::vector<PFCandidatePtr> deleteFrom(const PFCandidatePtr& ptr, const std::vector<PFCandidatePtr>& collection) 
  {
    std::vector<PFCandidatePtr> output;
    for ( std::vector<PFCandidatePtr>::const_iterator cand = collection.begin();
	  cand != collection.end(); ++cand ) {
      if ( (*cand) != ptr) output.push_back(*cand);
    }
    return output;
  }
}

class RecoTauTwoProngFilter : public RecoTauModifierPlugin {
  public:
  explicit RecoTauTwoProngFilter(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC);
    virtual ~RecoTauTwoProngFilter() {}
    void operator()(PFTau&) const override;
  private:
    double minPtFractionForSecondProng_;
};

  RecoTauTwoProngFilter::RecoTauTwoProngFilter(const edm::ParameterSet& pset, edm::ConsumesCollector &&iC):RecoTauModifierPlugin(pset,std::move(iC)) {
  minPtFractionForSecondProng_ = pset.getParameter<double>("minPtFractionForSecondProng");
}

void RecoTauTwoProngFilter::operator()(PFTau& tau) const {
  if (tau.signalPFChargedHadrCands().size() == 2) {
    const std::vector<PFCandidatePtr>& signalCharged = tau.signalPFChargedHadrCands();
    size_t indexOfHighestPt =
        (signalCharged[0]->pt() > signalCharged[1]->pt()) ? 0 : 1;
    size_t indexOfLowerPt   = ( indexOfHighestPt ) ? 0 : 1;
    double ratio = signalCharged[indexOfLowerPt]->pt()/
        signalCharged[indexOfHighestPt]->pt();

    if (ratio < minPtFractionForSecondProng_) {
      PFCandidatePtr keep = signalCharged[indexOfHighestPt];
      PFCandidatePtr filter = signalCharged[indexOfLowerPt];
      // Make our new signal charged candidate collection
      std::vector<PFCandidatePtr> newSignalCharged;
      newSignalCharged.push_back(keep);
      std::vector<PFCandidatePtr> newSignal = deleteFrom(filter, tau.signalPFCands());

      // Copy our filtered cand to isolation
      std::vector<PFCandidatePtr> newIsolationCharged =
          tau.isolationPFChargedHadrCands();
      newIsolationCharged.push_back(filter);
      std::vector<PFCandidatePtr> newIsolation = tau.isolationPFCands();
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

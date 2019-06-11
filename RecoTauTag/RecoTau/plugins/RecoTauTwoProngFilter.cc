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

namespace reco {
  namespace tau {

    namespace {
      // Delete an element from a ptr vector
      std::vector<CandidatePtr> deleteFrom(const CandidatePtr& ptr, const std::vector<CandidatePtr>& collection) {
        std::vector<CandidatePtr> output;
        for (std::vector<CandidatePtr>::const_iterator cand = collection.begin(); cand != collection.end(); ++cand) {
          if ((*cand) != ptr)
            output.push_back(*cand);
        }
        return output;
      }
    }  // namespace

    class RecoTauTwoProngFilter : public RecoTauModifierPlugin {
    public:
      explicit RecoTauTwoProngFilter(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);
      ~RecoTauTwoProngFilter() override {}
      void operator()(PFTau&) const override;

    private:
      double minPtFractionForSecondProng_;
    };

    RecoTauTwoProngFilter::RecoTauTwoProngFilter(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
        : RecoTauModifierPlugin(pset, std::move(iC)) {
      minPtFractionForSecondProng_ = pset.getParameter<double>("minPtFractionForSecondProng");
    }

    void RecoTauTwoProngFilter::operator()(PFTau& tau) const {
      if (tau.signalChargedHadrCands().size() == 2) {
        const std::vector<CandidatePtr>& signalCharged = tau.signalChargedHadrCands();
        size_t indexOfHighestPt = (signalCharged[0]->pt() > signalCharged[1]->pt()) ? 0 : 1;
        size_t indexOfLowerPt = (indexOfHighestPt) ? 0 : 1;
        double ratio = signalCharged[indexOfLowerPt]->pt() / signalCharged[indexOfHighestPt]->pt();

        if (ratio < minPtFractionForSecondProng_) {
          CandidatePtr keep = signalCharged[indexOfHighestPt];
          CandidatePtr filter = signalCharged[indexOfLowerPt];
          // Make our new signal charged candidate collection
          std::vector<CandidatePtr> newSignalCharged;
          newSignalCharged.push_back(keep);
          std::vector<CandidatePtr> newSignal = deleteFrom(filter, tau.signalCands());

          // Copy our filtered cand to isolation
          std::vector<CandidatePtr> newIsolationCharged = tau.isolationChargedHadrCands();
          newIsolationCharged.push_back(filter);
          std::vector<CandidatePtr> newIsolation = tau.isolationCands();
          newIsolation.push_back(filter);

          // Update tau members
          tau.setP4(tau.p4() - filter->p4());
          tau.setisolationPFChargedHadrCandsPtSum(tau.isolationPFChargedHadrCandsPtSum() - filter->pt());
          tau.setCharge(tau.charge() - filter->charge());
          // Update tau constituents
          tau.setsignalChargedHadrCands(newSignalCharged);
          tau.setsignalCands(newSignal);
          tau.setisolationChargedHadrCands(newIsolationCharged);
          tau.setisolationCands(newIsolation);
        }
      }
    }
  }  // namespace tau
}  // namespace reco
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory, reco::tau::RecoTauTwoProngFilter, "RecoTauTwoProngFilter");

/*
 * RecoTauPiZeroStripPlugin
 *
 * Merges PFGammas in a PFJet into Candidate piZeros defined as
 * strips in eta-phi.
 *
 * Author: Michail Bachtis (University of Wisconsin)
 *
 * Code modifications: Evan Friis (UC Davis)
 *
 */
#include <algorithm>
#include <functional>
#include <memory>

#include "RecoTauTag/RecoTau/interface/RecoTauPiZeroPlugins.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"

namespace reco {
  namespace tau {

    namespace {
      // Apply a hypothesis on the mass of the strips.
      math::XYZTLorentzVector applyMassConstraint(const math::XYZTLorentzVector& vec, double mass) {
        double factor = sqrt(vec.energy() * vec.energy() - mass * mass) / vec.P();
        return math::XYZTLorentzVector(vec.px() * factor, vec.py() * factor, vec.pz() * factor, vec.energy());
      }
    }  // namespace

    class RecoTauPiZeroStripPlugin : public RecoTauPiZeroBuilderPlugin {
    public:
      explicit RecoTauPiZeroStripPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);
      ~RecoTauPiZeroStripPlugin() override {}
      // Return type is unique_ptr<PiZeroVector>
      return_type operator()(const reco::Jet& jet) const override;
      // Hook to update PV information
      void beginEvent() override;

    private:
      std::unique_ptr<RecoTauQualityCuts> qcuts_;
      RecoTauVertexAssociator vertexAssociator_;

      std::vector<int> inputParticleIds_;  //type of candidates to clusterize
      double etaAssociationDistance_;      //eta Clustering Association Distance
      double phiAssociationDistance_;      //phi Clustering Association Distance

      // Parameters for build strip combinations
      bool combineStrips_;
      int maxStrips_;
      double combinatoricStripMassHypo_;

      AddFourMomenta p4Builder_;
    };

    RecoTauPiZeroStripPlugin::RecoTauPiZeroStripPlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
        : RecoTauPiZeroBuilderPlugin(pset, std::move(iC)),
          qcuts_(std::make_unique<RecoTauQualityCuts>(
              pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts"))),
          vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"), std::move(iC)) {
      inputParticleIds_ = pset.getParameter<std::vector<int> >("stripCandidatesParticleIds");
      etaAssociationDistance_ = pset.getParameter<double>("stripEtaAssociationDistance");
      phiAssociationDistance_ = pset.getParameter<double>("stripPhiAssociationDistance");
      combineStrips_ = pset.getParameter<bool>("makeCombinatoricStrips");
      if (combineStrips_) {
        maxStrips_ = pset.getParameter<int>("maxInputStrips");
        combinatoricStripMassHypo_ = pset.getParameter<double>("stripMassWhenCombining");
      }
    }

    // Update the primary vertex
    void RecoTauPiZeroStripPlugin::beginEvent() { vertexAssociator_.setEvent(*evt()); }

    RecoTauPiZeroStripPlugin::return_type RecoTauPiZeroStripPlugin::operator()(const reco::Jet& jet) const {
      // Get list of gamma candidates
      typedef std::vector<reco::CandidatePtr> CandPtrs;
      typedef CandPtrs::iterator CandIter;
      PiZeroVector output;

      // Get the candidates passing our quality cuts
      qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
      CandPtrs candsVector = qcuts_->filterCandRefs(pfCandidates(jet, inputParticleIds_));
      //PFCandPtrs candsVector = qcuts_->filterCandRefs(pfGammas(jet));

      // Convert to stl::list to allow fast deletions
      std::list<reco::CandidatePtr> cands;
      cands.insert(cands.end(), candsVector.begin(), candsVector.end());

      while (!cands.empty()) {
        // Seed this new strip, and delete it from future strips
        CandidatePtr seed = cands.front();
        cands.pop_front();

        // Add a new candidate to our collection using this seed
        auto strip = std::make_unique<RecoTauPiZero>(*seed, RecoTauPiZero::kStrips);
        strip->addDaughter(seed);

        // Find all other objects in the strip
        auto stripCand = cands.begin();
        while (stripCand != cands.end()) {
          if (fabs(strip->eta() - (*stripCand)->eta()) < etaAssociationDistance_ &&
              fabs(deltaPhi(*strip, **stripCand)) < phiAssociationDistance_) {
            // Add candidate to strip
            strip->addDaughter(*stripCand);
            // Update the strips four momenta
            p4Builder_.set(*strip);
            // Delete this candidate from future strips and move on to
            // the next potential candidate
            stripCand = cands.erase(stripCand);
          } else {
            // This candidate isn't compatabile - just move to the next candidate
            ++stripCand;
          }
        }
        // Update the vertex
        if (strip->daughterPtr(0).isNonnull())
          strip->setVertex(strip->daughterPtr(0)->vertex());
        output.emplace_back(strip.release());
      }

      // Check if we want to combine our strips
      if (combineStrips_ && output.size() > 1) {
        PiZeroVector stripCombinations;
        // Sort the output by descending pt
        std::sort(output.begin(), output.end(), [&](auto& arg1, auto& arg2) { return arg1->pt() > arg2->pt(); });
        // Get the end of interesting set of strips to try and combine
        PiZeroVector::const_iterator end_iter = takeNElements(output.begin(), output.end(), maxStrips_);

        // Look at all the combinations
        for (PiZeroVector::const_iterator firstIter = output.begin(); firstIter != end_iter - 1; ++firstIter) {
          for (PiZeroVector::const_iterator secondIter = firstIter + 1; secondIter != end_iter; ++secondIter) {
            auto const& first = *firstIter;
            auto const& second = *secondIter;
            Candidate::LorentzVector firstP4 = first->p4();
            Candidate::LorentzVector secondP4 = second->p4();
            // If we assume a certain mass for each strip apply it here.
            firstP4 = applyMassConstraint(firstP4, combinatoricStripMassHypo_);
            secondP4 = applyMassConstraint(secondP4, combinatoricStripMassHypo_);
            Candidate::LorentzVector totalP4 = firstP4 + secondP4;
            // Make our new combined strip
            auto combinedStrips =
                std::make_unique<RecoTauPiZero>(0,
                                                totalP4,
                                                Candidate::Point(0, 0, 0),
                                                //111, 10001, true, RecoTauPiZero::kCombinatoricStrips));
                                                111,
                                                10001,
                                                true,
                                                RecoTauPiZero::kUndefined);

            // Now loop over the strip members
            for (auto const& gamma : first->daughterPtrVector()) {
              combinedStrips->addDaughter(gamma);
            }
            for (auto const& gamma : second->daughterPtrVector()) {
              combinedStrips->addDaughter(gamma);
            }
            // Update the vertex
            if (combinedStrips->daughterPtr(0).isNonnull())
              combinedStrips->setVertex(combinedStrips->daughterPtr(0)->vertex());
            // Add to our collection of combined strips
            stripCombinations.emplace_back(combinedStrips.release());
          }
        }
        // When done doing all the combinations, add the combined strips to the output.
        std::move(stripCombinations.begin(), stripCombinations.end(), std::back_inserter(output));
      }

      return output;
    }
  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory, reco::tau::RecoTauPiZeroStripPlugin, "RecoTauPiZeroStripPlugin");

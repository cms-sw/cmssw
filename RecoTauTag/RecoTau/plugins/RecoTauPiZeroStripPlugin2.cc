/*
 * RecoTauPiZeroStripPlugin2
 *
 * Merges PFGammas in a PFJet into Candidate piZeros defined as
 * strips in eta-phi.
 *
 * Author: Michail Bachtis (University of Wisconsin)
 *
 * Code modifications: Evan Friis (UC Davis),
 *                     Christian Veelken (LLR)
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/CombinatoricGenerator.h"

//-------------------------------------------------------------------------------
// CV: the following headers are needed only for debug print-out
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"
//-------------------------------------------------------------------------------

namespace reco {
  namespace tau {

    namespace {
      // Apply a hypothesis on the mass of the strips.
      math::XYZTLorentzVector applyMassConstraint(const math::XYZTLorentzVector& vec, double mass) {
        double factor = sqrt(vec.energy() * vec.energy() - mass * mass) / vec.P();
        return math::XYZTLorentzVector(vec.px() * factor, vec.py() * factor, vec.pz() * factor, vec.energy());
      }
    }  // namespace

    class RecoTauPiZeroStripPlugin2 : public RecoTauPiZeroBuilderPlugin {
    public:
      explicit RecoTauPiZeroStripPlugin2(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
      ~RecoTauPiZeroStripPlugin2() override;
      // Return type is unique_ptr<PiZeroVector>
      return_type operator()(const reco::Jet&) const override;
      // Hook to update PV information
      void beginEvent() override;

    private:
      typedef std::vector<reco::CandidatePtr> CandPtrs;
      void addCandsToStrip(RecoTauPiZero&, CandPtrs&, const std::vector<bool>&, std::set<size_t>&, bool&) const;

      RecoTauVertexAssociator vertexAssociator_;

      std::unique_ptr<RecoTauQualityCuts> qcuts_;
      bool applyElecTrackQcuts_;
      double minGammaEtStripSeed_;
      double minGammaEtStripAdd_;

      double minStripEt_;

      std::vector<int> inputParticleIds_;  // type of candidates to clusterize
      double etaAssociationDistance_;      // size of strip clustering window in eta direction
      double phiAssociationDistance_;      // size of strip clustering window in phi direction

      bool updateStripAfterEachDaughter_;
      int maxStripBuildIterations_;

      // Parameters for build strip combinations
      bool combineStrips_;
      int maxStrips_;
      double combinatoricStripMassHypo_;

      AddFourMomenta p4Builder_;

      int verbosity_;
    };

    RecoTauPiZeroStripPlugin2::RecoTauPiZeroStripPlugin2(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
        : RecoTauPiZeroBuilderPlugin(pset, std::move(iC)),
          vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"), std::move(iC)),
          qcuts_(nullptr) {
      minGammaEtStripSeed_ = pset.getParameter<double>("minGammaEtStripSeed");
      minGammaEtStripAdd_ = pset.getParameter<double>("minGammaEtStripAdd");

      minStripEt_ = pset.getParameter<double>("minStripEt");

      edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
      //-------------------------------------------------------------------------------
      // CV: disable track quality cuts for PFElectronsPFElectron
      //       (treat PFElectrons like PFGammas for the purpose of building eta-phi strips)
      applyElecTrackQcuts_ = pset.getParameter<bool>("applyElecTrackQcuts");
      if (!applyElecTrackQcuts_) {
        qcuts_pset.addParameter<double>("minTrackPt", std::min(minGammaEtStripSeed_, minGammaEtStripAdd_));
        qcuts_pset.addParameter<double>("maxTrackChi2", 1.e+9);
        qcuts_pset.addParameter<double>("maxTransverseImpactParameter", 1.e+9);
        qcuts_pset.addParameter<double>("maxDeltaZ", 1.e+9);
        qcuts_pset.addParameter<double>("minTrackVertexWeight", -1.);
        qcuts_pset.addParameter<unsigned>("minTrackPixelHits", 0);
        qcuts_pset.addParameter<unsigned>("minTrackHits", 0);
      }
      //-------------------------------------------------------------------------------
      qcuts_pset.addParameter<double>("minGammaEt", std::min(minGammaEtStripSeed_, minGammaEtStripAdd_));
      qcuts_ = std::make_unique<RecoTauQualityCuts>(qcuts_pset);

      inputParticleIds_ = pset.getParameter<std::vector<int> >("stripCandidatesParticleIds");
      etaAssociationDistance_ = pset.getParameter<double>("stripEtaAssociationDistance");
      phiAssociationDistance_ = pset.getParameter<double>("stripPhiAssociationDistance");

      updateStripAfterEachDaughter_ = pset.getParameter<bool>("updateStripAfterEachDaughter");
      maxStripBuildIterations_ = pset.getParameter<int>("maxStripBuildIterations");

      combineStrips_ = pset.getParameter<bool>("makeCombinatoricStrips");
      if (combineStrips_) {
        maxStrips_ = pset.getParameter<int>("maxInputStrips");
        combinatoricStripMassHypo_ = pset.getParameter<double>("stripMassWhenCombining");
      }

      verbosity_ = pset.getParameter<int>("verbosity");
    }

    RecoTauPiZeroStripPlugin2::~RecoTauPiZeroStripPlugin2() {}

    // Update the primary vertex
    void RecoTauPiZeroStripPlugin2::beginEvent() { vertexAssociator_.setEvent(*evt()); }

    void RecoTauPiZeroStripPlugin2::addCandsToStrip(RecoTauPiZero& strip,
                                                    CandPtrs& cands,
                                                    const std::vector<bool>& candFlags,
                                                    std::set<size_t>& candIdsCurrentStrip,
                                                    bool& isCandAdded) const {
      if (verbosity_ >= 1) {
        edm::LogPrint("RecoTauPiZeroStripPlugin2") << "<RecoTauPiZeroStripPlugin2::addCandsToStrip>:";
      }
      size_t numCands = cands.size();
      for (size_t candId = 0; candId < numCands; ++candId) {
        if ((!candFlags[candId]) &&
            candIdsCurrentStrip.find(candId) == candIdsCurrentStrip.end()) {  // do not include same cand twice
          reco::CandidatePtr cand = cands[candId];
          if (fabs(strip.eta() - cand->eta()) <
                  etaAssociationDistance_ &&  // check if cand is within eta-phi window centered on strip
              fabs(strip.phi() - cand->phi()) < phiAssociationDistance_) {
            if (verbosity_ >= 2) {
              edm::LogPrint("RecoTauPiZeroStripPlugin2")
                  << "--> adding PFCand #" << candId << " (" << cand.id() << ":" << cand.key()
                  << "): Et = " << cand->et() << ", eta = " << cand->eta() << ", phi = " << cand->phi();
            }
            strip.addDaughter(cand);
            if (updateStripAfterEachDaughter_)
              p4Builder_.set(strip);
            isCandAdded = true;
            candIdsCurrentStrip.insert(candId);
          }
        }
      }
    }

    namespace {
      void markCandsInStrip(std::vector<bool>& candFlags, const std::set<size_t>& candIds) {
        for (std::set<size_t>::const_iterator candId = candIds.begin(); candId != candIds.end(); ++candId) {
          candFlags[*candId] = true;
        }
      }

      inline const reco::TrackBaseRef getTrack(const Candidate& cand) {
        const PFCandidate* pfCandPtr = dynamic_cast<const PFCandidate*>(&cand);
        if (pfCandPtr) {
          if (pfCandPtr->trackRef().isNonnull())
            return reco::TrackBaseRef(pfCandPtr->trackRef());
          else if (pfCandPtr->gsfTrackRef().isNonnull())
            return reco::TrackBaseRef(pfCandPtr->gsfTrackRef());
          else
            return reco::TrackBaseRef();
        }

        return reco::TrackBaseRef();
      }
    }  // namespace

    RecoTauPiZeroStripPlugin2::return_type RecoTauPiZeroStripPlugin2::operator()(const reco::Jet& jet) const {
      if (verbosity_ >= 1) {
        edm::LogPrint("RecoTauPiZeroStripPlugin2") << "<RecoTauPiZeroStripPlugin2::operator()>:";
        edm::LogPrint("RecoTauPiZeroStripPlugin2") << " minGammaEtStripSeed = " << minGammaEtStripSeed_;
        edm::LogPrint("RecoTauPiZeroStripPlugin2") << " minGammaEtStripAdd = " << minGammaEtStripAdd_;
        edm::LogPrint("RecoTauPiZeroStripPlugin2") << " minStripEt = " << minStripEt_;
      }

      PiZeroVector output;

      // Get the candidates passing our quality cuts
      qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
      CandPtrs candsVector = qcuts_->filterCandRefs(pfCandidates(jet, inputParticleIds_));

      // Convert to stl::list to allow fast deletions
      CandPtrs seedCands;
      CandPtrs addCands;
      int idx = 0;
      for (CandPtrs::iterator cand = candsVector.begin(); cand != candsVector.end(); ++cand) {
        if (verbosity_ >= 1) {
          edm::LogPrint("RecoTauPiZeroStripPlugin2")
              << "PFGamma #" << idx << " (" << cand->id() << ":" << cand->key() << "): Et = " << (*cand)->et()
              << ", eta = " << (*cand)->eta() << ", phi = " << (*cand)->phi();
        }
        if ((*cand)->et() > minGammaEtStripSeed_) {
          if (verbosity_ >= 2) {
            edm::LogPrint("RecoTauPiZeroStripPlugin2") << "--> assigning seedCandId = " << seedCands.size();
            const reco::TrackBaseRef candTrack = getTrack(**cand);
            if (candTrack.isNonnull()) {
              edm::LogPrint("RecoTauPiZeroStripPlugin2")
                  << "track: Pt = " << candTrack->pt() << " eta = " << candTrack->eta()
                  << ", phi = " << candTrack->phi() << ", charge = " << candTrack->charge();
              edm::LogPrint("RecoTauPiZeroStripPlugin2")
                  << " (dZ = " << candTrack->dz(vertexAssociator_.associatedVertex(jet)->position())
                  << ", dXY = " << candTrack->dxy(vertexAssociator_.associatedVertex(jet)->position()) << ","
                  << " numHits = " << candTrack->hitPattern().numberOfValidTrackerHits()
                  << ", numPxlHits = " << candTrack->hitPattern().numberOfValidPixelHits() << ","
                  << " chi2 = " << candTrack->normalizedChi2()
                  << ", dPt/Pt = " << (candTrack->ptError() / candTrack->pt()) << ")";
            }
          }
          seedCands.push_back(*cand);
        } else if ((*cand)->et() > minGammaEtStripAdd_) {
          if (verbosity_ >= 2) {
            edm::LogPrint("RecoTauPiZeroStripPlugin2") << "--> assigning addCandId = " << addCands.size();
          }
          addCands.push_back(*cand);
        }
        ++idx;
      }

      std::vector<bool> seedCandFlags(seedCands.size());  // true/false: seedCand is already/not yet included in strip
      std::vector<bool> addCandFlags(addCands.size());    // true/false: addCand  is already/not yet included in strip

      std::set<size_t> seedCandIdsCurrentStrip;
      std::set<size_t> addCandIdsCurrentStrip;

      size_t idxSeed = 0;
      while (idxSeed < seedCands.size()) {
        if (verbosity_ >= 2)
          edm::LogPrint("RecoTauPiZeroStripPlugin2") << "processing seed #" << idxSeed;

        seedCandIdsCurrentStrip.clear();
        addCandIdsCurrentStrip.clear();

        auto strip = std::make_unique<RecoTauPiZero>(*seedCands[idxSeed], RecoTauPiZero::kStrips);
        strip->addDaughter(seedCands[idxSeed]);
        seedCandIdsCurrentStrip.insert(idxSeed);

        bool isCandAdded;
        int stripBuildIteration = 0;
        do {
          isCandAdded = false;

          //if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin2") << " adding seedCands to strip..." ;
          addCandsToStrip(*strip, seedCands, seedCandFlags, seedCandIdsCurrentStrip, isCandAdded);
          //if ( verbosity_ >= 2 ) edm::LogPrint("RecoTauPiZeroStripPlugin2") << " adding addCands to strip..." ;
          addCandsToStrip(*strip, addCands, addCandFlags, addCandIdsCurrentStrip, isCandAdded);

          if (!updateStripAfterEachDaughter_)
            p4Builder_.set(*strip);

          ++stripBuildIteration;
        } while (isCandAdded && (stripBuildIteration < maxStripBuildIterations_ || maxStripBuildIterations_ == -1));

        if (strip->et() > minStripEt_) {  // strip passed Et cuts, add it to the event
          if (verbosity_ >= 2)
            edm::LogPrint("RecoTauPiZeroStripPlugin2")
                << "Building strip: Et = " << strip->et() << ", eta = " << strip->eta() << ", phi = " << strip->phi();

          // Update the vertex
          if (strip->daughterPtr(0).isNonnull())
            strip->setVertex(strip->daughterPtr(0)->vertex());
          output.push_back(std::move(strip));

          // Mark daughters as being part of this strip
          markCandsInStrip(seedCandFlags, seedCandIdsCurrentStrip);
          markCandsInStrip(addCandFlags, addCandIdsCurrentStrip);
        } else {  // strip failed Et cuts, just skip it
          if (verbosity_ >= 2)
            edm::LogPrint("RecoTauPiZeroStripPlugin2")
                << "Discarding strip: Et = " << strip->et() << ", eta = " << strip->eta() << ", phi = " << strip->phi();
        }

        ++idxSeed;
        while (idxSeed < seedCands.size() && seedCandFlags[idxSeed]) {
          ++idxSeed;  // fast-forward to next seed cand not yet included in any strip
        }
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
            stripCombinations.push_back(std::move(combinedStrips));
          }
        }
        // When done doing all the combinations, add the combined strips to the
        // output.
        std::move(stripCombinations.begin(), stripCombinations.end(), std::back_inserter(output));
      }

      return output;
    }
  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauPiZeroBuilderPluginFactory, reco::tau::RecoTauPiZeroStripPlugin2, "RecoTauPiZeroStripPlugin2");

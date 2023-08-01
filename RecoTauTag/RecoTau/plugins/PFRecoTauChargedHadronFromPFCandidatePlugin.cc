/*
 * PFRecoTauChargedHadronFromPFCandidatePlugin
 *
 * Build PFRecoTauChargedHadron objects
 * using charged PFCandidates and/or PFNeutralHadrons as input
 *
 * Author: Christian Veelken, LLR
 *
 */

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include <memory>

namespace reco {

  namespace tau {

    class PFRecoTauChargedHadronFromPFCandidatePlugin : public PFRecoTauChargedHadronBuilderPlugin {
    public:
      explicit PFRecoTauChargedHadronFromPFCandidatePlugin(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
      ~PFRecoTauChargedHadronFromPFCandidatePlugin() override;
      // Return type is unique_ptr<ChargedHadronVector>
      return_type operator()(const reco::Jet&) const override;
      // Hook to update PV information
      void beginEvent() override;

    private:
      typedef std::vector<reco::CandidatePtr> CandPtrs;

      RecoTauVertexAssociator vertexAssociator_;

      std::unique_ptr<RecoTauQualityCuts> qcuts_;

      std::vector<int> inputParticleIds_;  // type of candidates to clusterize

      double dRmergeNeutralHadronWrtChargedHadron_;
      double dRmergeNeutralHadronWrtNeutralHadron_;
      double dRmergeNeutralHadronWrtElectron_;
      double dRmergeNeutralHadronWrtOther_;
      int minBlockElementMatchesNeutralHadron_;
      int maxUnmatchedBlockElementsNeutralHadron_;
      double dRmergePhotonWrtChargedHadron_;
      double dRmergePhotonWrtNeutralHadron_;
      double dRmergePhotonWrtElectron_;
      double dRmergePhotonWrtOther_;
      int minBlockElementMatchesPhoton_;
      int maxUnmatchedBlockElementsPhoton_;
      double minMergeNeutralHadronEt_;
      double minMergeGammaEt_;
      double minMergeChargedHadronPt_;

      const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bFieldToken_;
      double bField_;

      int verbosity_;
    };

    PFRecoTauChargedHadronFromPFCandidatePlugin::PFRecoTauChargedHadronFromPFCandidatePlugin(
        const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
        : PFRecoTauChargedHadronBuilderPlugin(pset, std::move(iC)),
          vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"), std::move(iC)),
          qcuts_(nullptr),
          bFieldToken_(iC.esConsumes()) {
      edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
      qcuts_ = std::make_unique<RecoTauQualityCuts>(qcuts_pset);

      inputParticleIds_ = pset.getParameter<std::vector<int> >("chargedHadronCandidatesParticleIds");

      dRmergeNeutralHadronWrtChargedHadron_ = pset.getParameter<double>("dRmergeNeutralHadronWrtChargedHadron");
      dRmergeNeutralHadronWrtNeutralHadron_ = pset.getParameter<double>("dRmergeNeutralHadronWrtNeutralHadron");
      dRmergeNeutralHadronWrtElectron_ = pset.getParameter<double>("dRmergeNeutralHadronWrtElectron");
      dRmergeNeutralHadronWrtOther_ = pset.getParameter<double>("dRmergeNeutralHadronWrtOther");
      minBlockElementMatchesNeutralHadron_ = pset.getParameter<int>("minBlockElementMatchesNeutralHadron");
      maxUnmatchedBlockElementsNeutralHadron_ = pset.getParameter<int>("maxUnmatchedBlockElementsNeutralHadron");
      dRmergePhotonWrtChargedHadron_ = pset.getParameter<double>("dRmergePhotonWrtChargedHadron");
      dRmergePhotonWrtNeutralHadron_ = pset.getParameter<double>("dRmergePhotonWrtNeutralHadron");
      dRmergePhotonWrtElectron_ = pset.getParameter<double>("dRmergePhotonWrtElectron");
      dRmergePhotonWrtOther_ = pset.getParameter<double>("dRmergePhotonWrtOther");
      minBlockElementMatchesPhoton_ = pset.getParameter<int>("minBlockElementMatchesPhoton");
      maxUnmatchedBlockElementsPhoton_ = pset.getParameter<int>("maxUnmatchedBlockElementsPhoton");
      minMergeNeutralHadronEt_ = pset.getParameter<double>("minMergeNeutralHadronEt");
      minMergeGammaEt_ = pset.getParameter<double>("minMergeGammaEt");
      minMergeChargedHadronPt_ = pset.getParameter<double>("minMergeChargedHadronPt");

      verbosity_ = pset.getParameter<int>("verbosity");
    }

    PFRecoTauChargedHadronFromPFCandidatePlugin::~PFRecoTauChargedHadronFromPFCandidatePlugin() {}

    // Update the primary vertex
    void PFRecoTauChargedHadronFromPFCandidatePlugin::beginEvent() {
      vertexAssociator_.setEvent(*evt());

      bField_ = evtSetup()->getData(bFieldToken_).inTesla(GlobalPoint(0, 0, 0)).z();
    }

    namespace {
      bool isMatchedByBlockElement(const reco::PFCandidate& pfCandidate1,
                                   const reco::PFCandidate& pfCandidate2,
                                   int minMatches1,
                                   int minMatches2,
                                   int maxUnmatchedBlockElements1plus2) {
        reco::PFCandidate::ElementsInBlocks blockElements1 = pfCandidate1.elementsInBlocks();
        int numBlocks1 = blockElements1.size();
        reco::PFCandidate::ElementsInBlocks blockElements2 = pfCandidate2.elementsInBlocks();
        int numBlocks2 = blockElements2.size();
        int numBlocks_matched = 0;
        for (reco::PFCandidate::ElementsInBlocks::const_iterator blockElement1 = blockElements1.begin();
             blockElement1 != blockElements1.end();
             ++blockElement1) {
          bool isMatched = false;
          for (reco::PFCandidate::ElementsInBlocks::const_iterator blockElement2 = blockElements2.begin();
               blockElement2 != blockElements2.end();
               ++blockElement2) {
            if (blockElement1->first.id() == blockElement2->first.id() &&
                blockElement1->first.key() == blockElement2->first.key() &&
                blockElement1->second == blockElement2->second) {
              isMatched = true;
            }
          }
          if (isMatched)
            ++numBlocks_matched;
        }
        assert(numBlocks_matched <= numBlocks1);
        assert(numBlocks_matched <= numBlocks2);
        if (numBlocks_matched >= minMatches1 && numBlocks_matched >= minMatches2 &&
            ((numBlocks1 - numBlocks_matched) + (numBlocks2 - numBlocks_matched)) <= maxUnmatchedBlockElements1plus2) {
          return true;
        } else {
          return false;
        }
      }
    }  // namespace

    PFRecoTauChargedHadronFromPFCandidatePlugin::return_type PFRecoTauChargedHadronFromPFCandidatePlugin::operator()(
        const reco::Jet& jet) const {
      if (verbosity_) {
        edm::LogPrint("TauChHadronFromPF") << "<PFRecoTauChargedHadronFromPFCandidatePlugin::operator()>:";
        edm::LogPrint("TauChHadronFromPF") << " pluginName = " << name();
      }

      ChargedHadronVector output;

      // Get the candidates passing our quality cuts
      qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
      CandPtrs candsVector = qcuts_->filterCandRefs(pfCandidates(jet, inputParticleIds_));

      for (CandPtrs::iterator cand = candsVector.begin(); cand != candsVector.end(); ++cand) {
        if (verbosity_) {
          edm::LogPrint("TauChHadronFromPF")
              << "processing PFCandidate: Pt = " << (*cand)->pt() << ", eta = " << (*cand)->eta()
              << ", phi = " << (*cand)->phi() << " (pdgId = " << (*cand)->pdgId() << ", charge = " << (*cand)->charge()
              << ")";
        }

        PFRecoTauChargedHadron::PFRecoTauChargedHadronAlgorithm algo = PFRecoTauChargedHadron::kUndefined;
        if (std::abs((*cand)->charge()) > 0.5)
          algo = PFRecoTauChargedHadron::kChargedPFCandidate;
        else
          algo = PFRecoTauChargedHadron::kPFNeutralHadron;
        auto chargedHadron = std::make_unique<PFRecoTauChargedHadron>(**cand, algo);

        const reco::PFCandidate* pfCand = dynamic_cast<const reco::PFCandidate*>(&**cand);
        if (pfCand) {
          if (pfCand->trackRef().isNonnull())
            chargedHadron->track_ = edm::refToPtr(pfCand->trackRef());
          else if (pfCand->muonRef().isNonnull() && pfCand->muonRef()->innerTrack().isNonnull())
            chargedHadron->track_ = edm::refToPtr(pfCand->muonRef()->innerTrack());
          else if (pfCand->muonRef().isNonnull() && pfCand->muonRef()->globalTrack().isNonnull())
            chargedHadron->track_ = edm::refToPtr(pfCand->muonRef()->globalTrack());
          else if (pfCand->muonRef().isNonnull() && pfCand->muonRef()->outerTrack().isNonnull())
            chargedHadron->track_ = edm::refToPtr(pfCand->muonRef()->outerTrack());
          else if (pfCand->gsfTrackRef().isNonnull())
            chargedHadron->track_ = edm::refToPtr(pfCand->gsfTrackRef());
        }  // TauReco@MiniAOD: Tracks only available dynamically, so no possiblity to save ref here; checked by code downstream

        chargedHadron->positionAtECALEntrance_ = atECALEntrance(&**cand, bField_);
        chargedHadron->chargedPFCandidate_ = (*cand);
        chargedHadron->addDaughter(*cand);

        int pdgId = std::abs(chargedHadron->chargedPFCandidate_->pdgId());

        if (chargedHadron->pt() > minMergeChargedHadronPt_) {
          for (const auto& jetConstituent : jet.daughterPtrVector()) {
            // CV: take care of not double-counting energy in case "charged" PFCandidate is in fact a PFNeutralHadron
            if (jetConstituent == chargedHadron->chargedPFCandidate_)
              continue;

            int jetConstituentPdgId = std::abs(jetConstituent->pdgId());
            if (!(jetConstituentPdgId == 130 || jetConstituentPdgId == 22))
              continue;

            double dR = deltaR(atECALEntrance(jetConstituent.get(), bField_),
                               atECALEntrance(chargedHadron->chargedPFCandidate_.get(), bField_));
            double dRmerge = -1.;
            int minBlockElementMatches = 1000;
            int maxUnmatchedBlockElements = 0;
            double minMergeEt = 1.e+6;
            if (jetConstituentPdgId == 130) {
              if (pdgId == 211)
                dRmerge = dRmergeNeutralHadronWrtChargedHadron_;
              else if (pdgId == 130)
                dRmerge = dRmergeNeutralHadronWrtNeutralHadron_;
              else if (pdgId == 11)
                dRmerge = dRmergeNeutralHadronWrtElectron_;
              else
                dRmerge = dRmergeNeutralHadronWrtOther_;
              minBlockElementMatches = minBlockElementMatchesNeutralHadron_;
              maxUnmatchedBlockElements = maxUnmatchedBlockElementsNeutralHadron_;
              minMergeEt = minMergeNeutralHadronEt_;
            } else if (jetConstituentPdgId == 22) {
              if (pdgId == 211)
                dRmerge = dRmergePhotonWrtChargedHadron_;
              else if (pdgId == 130)
                dRmerge = dRmergePhotonWrtNeutralHadron_;
              else if (pdgId == 11)
                dRmerge = dRmergePhotonWrtElectron_;
              else
                dRmerge = dRmergePhotonWrtOther_;
              minBlockElementMatches = minBlockElementMatchesPhoton_;
              maxUnmatchedBlockElements = maxUnmatchedBlockElementsPhoton_;
              minMergeEt = minMergeGammaEt_;
            }

            if (jetConstituent->et() > minMergeEt) {
              if (dR < dRmerge) {
                chargedHadron->neutralPFCandidates_.push_back(jetConstituent);
                chargedHadron->addDaughter(jetConstituent);
              } else {
                // TauReco@MiniAOD: No access to PF blocks at MiniAOD level, but the code below seems to have very minor impact
                const reco::PFCandidate* pfJetConstituent =
                    dynamic_cast<const reco::PFCandidate*>(jetConstituent.get());
                if (pfCand != nullptr && pfJetConstituent != nullptr) {
                  if (isMatchedByBlockElement(*pfJetConstituent,
                                              *pfCand,
                                              minBlockElementMatches,
                                              minBlockElementMatches,
                                              maxUnmatchedBlockElements)) {
                    chargedHadron->neutralPFCandidates_.push_back(jetConstituent);
                    chargedHadron->addDaughter(jetConstituent);
                  }
                }
              }
            }
          }
        }

        setChargedHadronP4(*chargedHadron);

        if (verbosity_) {
          edm::LogPrint("TauChHadronFromPF") << *chargedHadron;
        }
        // Update the vertex
        if (chargedHadron->daughterPtr(0).isNonnull())
          chargedHadron->setVertex(chargedHadron->daughterPtr(0)->vertex());
        output.push_back(std::move(chargedHadron));
      }

      return output;
    }

  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronBuilderPluginFactory,
                  reco::tau::PFRecoTauChargedHadronFromPFCandidatePlugin,
                  "PFRecoTauChargedHadronFromPFCandidatePlugin");

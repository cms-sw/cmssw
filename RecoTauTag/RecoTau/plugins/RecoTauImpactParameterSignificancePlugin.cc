/*
 * =============================================================================
 *       Filename:  RecoTauImpactParameterSignificancePlugin.cc
 *
 *    Description:  Add the IP significance of the lead track w.r.t to the PV.
 *                  to a PFTau.
 *        Created:  10/31/2010 13:32:14
 *
 *         Authors:  Evan K. Friis (UC Davis), evan.klose.friis@cern.ch,
 *                   Simone Gennai, Ludovic Houchu
 *
 * =============================================================================
 */

#include "RecoTauTag/RecoTau/interface/RecoTauBuilderPlugins.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

namespace reco {
  namespace tau {

    class RecoTauImpactParameterSignificancePlugin : public RecoTauModifierPlugin {
    public:
      explicit RecoTauImpactParameterSignificancePlugin(const edm::ParameterSet& pset, edm::ConsumesCollector&& iC);
      ~RecoTauImpactParameterSignificancePlugin() override {}
      void operator()(PFTau& tau) const override;
      void beginEvent() override;

    private:
      const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transTrackBuilderToken_;
      RecoTauVertexAssociator vertexAssociator_;
      const TransientTrackBuilder* builder_;
    };

    RecoTauImpactParameterSignificancePlugin ::RecoTauImpactParameterSignificancePlugin(const edm::ParameterSet& pset,
                                                                                        edm::ConsumesCollector&& iC)
        : RecoTauModifierPlugin(pset, std::move(iC)),
          transTrackBuilderToken_(iC.esConsumes(edm::ESInputTag{"", "TransientTrackBuilder"})),
          vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"), std::move(iC)) {}

    void RecoTauImpactParameterSignificancePlugin::beginEvent() {
      vertexAssociator_.setEvent(*evt());
      // Get tranisent track builder.
      builder_ = &evtSetup()->getData(transTrackBuilderToken_);
    }

    namespace {
      inline const reco::Track* getTrack(const Candidate& cand) {
        const PFCandidate* pfCandPtr = dynamic_cast<const PFCandidate*>(&cand);
        if (pfCandPtr) {
          if (pfCandPtr->trackRef().isNonnull())
            return pfCandPtr->trackRef().get();
          else
            return nullptr;
        }

        const pat::PackedCandidate* packedCand = dynamic_cast<const pat::PackedCandidate*>(&cand);
        if (packedCand && packedCand->hasTrackDetails())
          return &packedCand->pseudoTrack();

        return nullptr;
      }
    }  // namespace

    void RecoTauImpactParameterSignificancePlugin::operator()(PFTau& tau) const {
      // Get the transient lead track
      if (tau.leadChargedHadrCand().isNonnull()) {
        const reco::Track* leadTrack = getTrack(*tau.leadChargedHadrCand());
        if (leadTrack != nullptr) {
          const TransientTrack track = builder_->build(leadTrack);
          GlobalVector direction(tau.jetRef()->px(), tau.jetRef()->py(), tau.jetRef()->pz());
          VertexRef pv = vertexAssociator_.associatedVertex(tau);
          // Compute the significance
          std::pair<bool, Measurement1D> ipsig = IPTools::signedImpactParameter3D(track, direction, *pv);
          if (ipsig.first)
            tau.setleadPFChargedHadrCandsignedSipt(ipsig.second.significance());
        }
      }
    }

  }  // namespace tau
}  // namespace reco
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_EDM_PLUGIN(RecoTauModifierPluginFactory,
                  reco::tau::RecoTauImpactParameterSignificancePlugin,
                  "RecoTauImpactParameterSignificancePlugin");

/*
 * PFRecoTauChargedHadronFromGenericTrackPlugin
 *
 * Build PFRecoTauChargedHadron objects
 * using tracks as input, from either collection of RECO/AOD reco::Tracks 
 * (PFRecoTauChargedHadronFromTrackPlugin) or from a collection of MINIAOD
 * pat::PackedCandidates (PFRecoTauChargedHadronFromLostTrackPlugin), typically
 * using the 'lostTracks' collection
 *
 * Author: Christian Veelken, LLR
 *
 * inclusion of lost tracks based on original implementation
 * by Michal Bluj, NCBJ, Poland
 */

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/BaseParticlePropagator/interface/BaseParticlePropagator.h"
#include "CommonTools/BaseParticlePropagator/interface/RawParticle.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/RecoTauQualityCuts.h"
#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"
#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include <TMath.h>

#include <memory>
#include <cmath>
#include <algorithm>
#include <atomic>

namespace reco {
  namespace tau {

    template <class TrackClass>
    class PFRecoTauChargedHadronFromGenericTrackPlugin : public PFRecoTauChargedHadronBuilderPlugin {
    public:
      explicit PFRecoTauChargedHadronFromGenericTrackPlugin(const edm::ParameterSet&, edm::ConsumesCollector&& iC);
      ~PFRecoTauChargedHadronFromGenericTrackPlugin() override;
      // Return type is unique_ptr<ChargedHadronVector>
      return_type operator()(const reco::Jet&) const override;
      // Hook to update PV information
      void beginEvent() override;

    private:
      bool filterTrack(const edm::Handle<std::vector<TrackClass> >&, size_t iTrack) const;
      void setChargedHadronTrack(PFRecoTauChargedHadron& chargedHadron, const edm::Ptr<TrackClass>& track) const;
      double getTrackPtError(const TrackClass& track) const;
      XYZTLorentzVector getTrackPos(const TrackClass& track) const;

      RecoTauVertexAssociator vertexAssociator_;

      std::unique_ptr<RecoTauQualityCuts> qcuts_;

      edm::InputTag srcTracks_;
      edm::EDGetTokenT<std::vector<TrackClass> > Tracks_token;
      const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
      double dRcone_;
      bool dRconeLimitedToJetArea_;

      double dRmergeNeutralHadron_;
      double dRmergePhoton_;

      math::XYZVector magneticFieldStrength_;

      static std::atomic<unsigned int> numWarnings_;
      static constexpr unsigned int maxWarnings_ = 3;

      int verbosity_;
    };

    template <class TrackClass>
    std::atomic<unsigned int> PFRecoTauChargedHadronFromGenericTrackPlugin<TrackClass>::numWarnings_{0};

    template <class TrackClass>
    PFRecoTauChargedHadronFromGenericTrackPlugin<TrackClass>::PFRecoTauChargedHadronFromGenericTrackPlugin(
        const edm::ParameterSet& pset, edm::ConsumesCollector&& iC)
        : PFRecoTauChargedHadronBuilderPlugin(pset, std::move(iC)),
          vertexAssociator_(pset.getParameter<edm::ParameterSet>("qualityCuts"), std::move(iC)),
          qcuts_(nullptr),
          magneticFieldToken_(iC.esConsumes()) {
      edm::ParameterSet qcuts_pset = pset.getParameterSet("qualityCuts").getParameterSet("signalQualityCuts");
      qcuts_ = std::make_unique<RecoTauQualityCuts>(qcuts_pset);

      srcTracks_ = pset.getParameter<edm::InputTag>("srcTracks");
      Tracks_token = iC.consumes<std::vector<TrackClass> >(srcTracks_);
      dRcone_ = pset.getParameter<double>("dRcone");
      dRconeLimitedToJetArea_ = pset.getParameter<bool>("dRconeLimitedToJetArea");

      dRmergeNeutralHadron_ = pset.getParameter<double>("dRmergeNeutralHadron");
      dRmergePhoton_ = pset.getParameter<double>("dRmergePhoton");

      verbosity_ = pset.getParameter<int>("verbosity");
    }

    template <class TrackClass>
    PFRecoTauChargedHadronFromGenericTrackPlugin<TrackClass>::~PFRecoTauChargedHadronFromGenericTrackPlugin() {}

    // Update the primary vertex
    template <class TrackClass>
    void PFRecoTauChargedHadronFromGenericTrackPlugin<TrackClass>::beginEvent() {
      vertexAssociator_.setEvent(*this->evt());

      magneticFieldStrength_ = evtSetup()->getData(magneticFieldToken_).inTesla(GlobalPoint(0., 0., 0.));
    }

    template <>
    bool PFRecoTauChargedHadronFromGenericTrackPlugin<reco::Track>::filterTrack(
        const edm::Handle<std::vector<reco::Track> >& tracks, size_t iTrack) const {
      // ignore tracks which fail quality cuts
      reco::TrackRef trackRef(tracks, iTrack);
      return qcuts_->filterTrack(trackRef);
    }

    template <>
    bool PFRecoTauChargedHadronFromGenericTrackPlugin<pat::PackedCandidate>::filterTrack(
        const edm::Handle<std::vector<pat::PackedCandidate> >& tracks, size_t iTrack) const {
      // ignore tracks which fail quality cuts
      const pat::PackedCandidate& cand = (*tracks)[iTrack];
      if (cand.charge() == 0)
        return false;

      return qcuts_->filterChargedCand(cand);
    }

    template <>
    void PFRecoTauChargedHadronFromGenericTrackPlugin<reco::Track>::setChargedHadronTrack(
        PFRecoTauChargedHadron& chargedHadron, const edm::Ptr<reco::Track>& track) const {
      chargedHadron.track_ = track;
    }

    template <>
    void PFRecoTauChargedHadronFromGenericTrackPlugin<pat::PackedCandidate>::setChargedHadronTrack(
        PFRecoTauChargedHadron& chargedHadron, const edm::Ptr<pat::PackedCandidate>& track) const {
      chargedHadron.lostTrackCandidate_ = track;
    }

    template <>
    double PFRecoTauChargedHadronFromGenericTrackPlugin<reco::Track>::getTrackPtError(const reco::Track& track) const {
      return track.ptError();
    }

    template <>
    double PFRecoTauChargedHadronFromGenericTrackPlugin<pat::PackedCandidate>::getTrackPtError(
        const pat::PackedCandidate& cand) const {
      double trackPtError =
          0.06;  // MB: Approximate avarage track PtError by 2.5% (barrel), 4% (transition), 6% (endcaps) lostTracks w/o detailed track information available (after TRK-11-001)
      const reco::Track* track(cand.bestTrack());
      if (track != nullptr) {
        trackPtError = track->ptError();
      } else {
        if (std::abs(cand.eta()) < 0.9)
          trackPtError = 0.025;
        else if (std::abs(cand.eta()) < 1.4)
          trackPtError = 0.04;
      }
      return trackPtError;
    }

    template <>
    XYZTLorentzVector PFRecoTauChargedHadronFromGenericTrackPlugin<reco::Track>::getTrackPos(
        const reco::Track& track) const {
      return XYZTLorentzVector(track.referencePoint().x(), track.referencePoint().y(), track.referencePoint().z(), 0.);
    }

    template <>
    XYZTLorentzVector PFRecoTauChargedHadronFromGenericTrackPlugin<pat::PackedCandidate>::getTrackPos(
        const pat::PackedCandidate& track) const {
      return XYZTLorentzVector(track.vertex().x(), track.vertex().y(), track.vertex().z(), 0.);
    }

    namespace {
      struct Candidate_withDistance {
        reco::CandidatePtr pfCandidate_;
        double distance_;
      };

      bool isSmallerDistance(const Candidate_withDistance& cand1, const Candidate_withDistance& cand2) {
        return (cand1.distance_ < cand2.distance_);
      }
    }  // namespace

    template <class TrackClass>
    typename PFRecoTauChargedHadronFromGenericTrackPlugin<TrackClass>::return_type
    PFRecoTauChargedHadronFromGenericTrackPlugin<TrackClass>::operator()(const reco::Jet& jet) const {
      if (verbosity_) {
        edm::LogPrint("TauChHFromTrack") << "<PFRecoTauChargedHadronFromGenericTrackPlugin::operator()>:";
        edm::LogPrint("TauChHFromTrack") << " pluginName = " << name();
      }

      ChargedHadronVector output;

      const edm::Event& evt = (*this->evt());

      edm::Handle<std::vector<TrackClass> > tracks;
      evt.getByToken(Tracks_token, tracks);

      qcuts_->setPV(vertexAssociator_.associatedVertex(jet));
      float jEta = jet.eta();
      float jPhi = jet.phi();
      size_t numTracks = tracks->size();
      for (size_t iTrack = 0; iTrack < numTracks; ++iTrack) {
        const TrackClass& track = (*tracks)[iTrack];

        // consider tracks in vicinity of tau-jet candidate only
        double dR = deltaR(track.eta(), track.phi(), jEta, jPhi);
        double dRmatch = dRcone_;
        if (dRconeLimitedToJetArea_) {
          double jetArea = jet.jetArea();
          if (jetArea > 0.) {
            dRmatch = std::min(dRmatch, sqrt(jetArea / M_PI));
          } else {
            if (numWarnings_ < maxWarnings_) {
              edm::LogInfo("PFRecoTauChargedHadronFromGenericTrackPlugin::operator()")
                  << "Jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi()
                  << " has area = " << jetArea << " !!" << std::endl;
              ++numWarnings_;
            }
            dRmatch = 0.1;
          }
        }
        if (dR > dRmatch)
          continue;

        if (!this->filterTrack(tracks, iTrack))
          continue;

        reco::Candidate::Charge trackCharge_int = 0;
        if (track.charge() > 0.)
          trackCharge_int = +1;
        else if (track.charge() < 0.)
          trackCharge_int = -1;

        const double chargedPionMass = 0.13957;  // GeV
        double chargedPionP = track.p();
        double chargedPionEn = TMath::Sqrt(chargedPionP * chargedPionP + chargedPionMass * chargedPionMass);
        reco::Candidate::LorentzVector chargedPionP4(track.px(), track.py(), track.pz(), chargedPionEn);

        reco::Vertex::Point vtx(0., 0., 0.);
        if (vertexAssociator_.associatedVertex(jet).isNonnull())
          vtx = vertexAssociator_.associatedVertex(jet)->position();

        auto chargedHadron = std::make_unique<PFRecoTauChargedHadron>(
            trackCharge_int, chargedPionP4, vtx, 0, true, PFRecoTauChargedHadron::kTrack);

        setChargedHadronTrack(*chargedHadron, edm::Ptr<TrackClass>(tracks, iTrack));

        // CV: Take code for propagating track to ECAL entrance
        //     from RecoParticleFlow/PFTracking/src/PFTrackTransformer.cc
        //     to make sure propagation is done in the same way as for charged PFCandidates.
        //
        //     The following replacements need to be made
        //       outerMomentum -> momentum
        //       outerPosition -> referencePoint
        //     in order to run on AOD input
        //    (outerMomentum and outerPosition require access to reco::TrackExtra objects, which are available in RECO only)
        //
        XYZTLorentzVector chargedPionPos(getTrackPos(track));
        RawParticle p(chargedPionP4, chargedPionPos);
        p.setCharge(track.charge());
        BaseParticlePropagator trackPropagator(p, 0., 0., magneticFieldStrength_.z());
        trackPropagator.propagateToEcalEntrance(false);
        if (trackPropagator.getSuccess() != 0) {
          chargedHadron->positionAtECALEntrance_ = trackPropagator.particle().vertex();
        } else {
          if (chargedPionP4.pt() > 2. and std::abs(chargedPionP4.eta()) < 3.) {
            edm::LogWarning("PFRecoTauChargedHadronFromGenericTrackPlugin::operator()")
                << "Failed to propagate track: Pt = " << track.pt() << ", eta = " << track.eta()
                << ", phi = " << track.phi() << " to ECAL entrance !!" << std::endl;
          }
          chargedHadron->positionAtECALEntrance_ = math::XYZPointF(0., 0., 0.);
        }

        std::vector<Candidate_withDistance> neutralJetConstituents_withDistance;
        for (const auto& jetConstituent : jet.daughterPtrVector()) {
          int pdgId = jetConstituent->pdgId();
          if (!(pdgId == 130 || pdgId == 22))
            continue;
          double dR = deltaR(atECALEntrance(&*jetConstituent, magneticFieldStrength_.z()),
                             chargedHadron->positionAtECALEntrance_);
          double dRmerge = -1.;
          if (pdgId == 130)
            dRmerge = dRmergeNeutralHadron_;
          else if (pdgId == 22)
            dRmerge = dRmergePhoton_;
          if (dR < dRmerge) {
            Candidate_withDistance jetConstituent_withDistance;
            jetConstituent_withDistance.pfCandidate_ = jetConstituent;
            jetConstituent_withDistance.distance_ = dR;
            neutralJetConstituents_withDistance.push_back(jetConstituent_withDistance);
            chargedHadron->addDaughter(jetConstituent);
          }
        }
        std::sort(
            neutralJetConstituents_withDistance.begin(), neutralJetConstituents_withDistance.end(), isSmallerDistance);

        const double caloResolutionCoeff =
            1.0;  // CV: approximate ECAL + HCAL calorimeter resolution for hadrons by 100%*sqrt(E)
        double resolutionTrackP = track.p() * (getTrackPtError(track) / track.pt());
        double neutralEnSum = 0.;
        for (std::vector<Candidate_withDistance>::const_iterator nextNeutral =
                 neutralJetConstituents_withDistance.begin();
             nextNeutral != neutralJetConstituents_withDistance.end();
             ++nextNeutral) {
          double nextNeutralEn = nextNeutral->pfCandidate_->energy();
          double resolutionCaloEn = caloResolutionCoeff * sqrt(neutralEnSum + nextNeutralEn);
          double resolution = sqrt(resolutionTrackP * resolutionTrackP + resolutionCaloEn * resolutionCaloEn);
          if ((neutralEnSum + nextNeutralEn) < (track.p() + 2. * resolution)) {
            chargedHadron->neutralPFCandidates_.push_back(nextNeutral->pfCandidate_);
            neutralEnSum += nextNeutralEn;
          } else {
            break;
          }
        }

        setChargedHadronP4(*chargedHadron);

        if (verbosity_) {
          edm::LogPrint("TauChHFromTrack") << *chargedHadron;
        }

        output.push_back(std::move(chargedHadron));
      }

      return output;
    }

    typedef PFRecoTauChargedHadronFromGenericTrackPlugin<reco::Track> PFRecoTauChargedHadronFromTrackPlugin;
    typedef PFRecoTauChargedHadronFromGenericTrackPlugin<pat::PackedCandidate> PFRecoTauChargedHadronFromLostTrackPlugin;

  }  // namespace tau
}  // namespace reco

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronBuilderPluginFactory,
                  reco::tau::PFRecoTauChargedHadronFromTrackPlugin,
                  "PFRecoTauChargedHadronFromTrackPlugin");
DEFINE_EDM_PLUGIN(PFRecoTauChargedHadronBuilderPluginFactory,
                  reco::tau::PFRecoTauChargedHadronFromLostTrackPlugin,
                  "PFRecoTauChargedHadronFromLostTrackPlugin");

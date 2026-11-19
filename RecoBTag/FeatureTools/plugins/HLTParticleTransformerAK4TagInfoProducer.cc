#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/BTauReco/interface/HLTParticleTransformerAK4Features.h"
#include "DataFormats/BTauReco/interface/HLTParticleTransformerAK4TagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/IPTools/interface/IPTools.h"

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

#include <algorithm>
#include <array>
#include <vector>
#include <cmath>

#include "TVector3.h"

class HLTParticleTransformerAK4TagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTParticleTransformerAK4TagInfoProducer(const edm::ParameterSet&);
  ~HLTParticleTransformerAK4TagInfoProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::vector<reco::HLTParticleTransformerAK4TagInfo> TagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  // Find a persistent edm::Ref to the candidate in the original collection.
  edm::Ref<edm::View<reco::Candidate>> getPersistentCandidate(
      const reco::Candidate* cand, const edm::Handle<edm::View<reco::Candidate>>& handle) const {
    for (size_t idx = 0; idx < handle->size(); ++idx) {
      if (&(handle->at(idx)) == cand) {
        return edm::Ref<edm::View<reco::Candidate>>(handle, idx);
      }
    }
    return edm::Ref<edm::View<reco::Candidate>>();
  }

  // Build a PackedCandidate with packed track properties.
  static pat::PackedCandidate buildPackedCandidate(const reco::PFCandidate* cand,
                                                   const reco::Track* track,
                                                   int pvAssocQual,
                                                   const reco::VertexRef& pv_ass,
                                                   const reco::VertexRefProd& pvRefProd) {
    constexpr float min_track_pt_property = 0.5f;
    constexpr int min_valid_pixel_hits = 0;
    constexpr int covarianceVersion = 1;
    constexpr std::array<int, 5> covariancePackingSchemas = {{8, 264, 520, 776, 0}};

    pat::PackedCandidate packed;
    if (track) {
      packed = pat::PackedCandidate(cand->polarP4(),
                                    track->referencePoint(),
                                    track->pt(),
                                    track->eta(),
                                    track->phi(),
                                    cand->pdgId(),
                                    pvRefProd,
                                    pv_ass.key());
      packed.setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(pvAssocQual));
      packed.setCovarianceVersion(covarianceVersion);

      pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
      int nlost = track->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      if (nlost == 0) {
        if (track->hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1))
          lostHits = pat::PackedCandidate::validHitInFirstPixelBarrelLayer;
      } else {
        lostHits = (nlost == 1 ? pat::PackedCandidate::oneLostInnerHit : pat::PackedCandidate::moreLostInnerHits);
      }
      packed.setLostInnerHits(lostHits);
      packed.setTrkAlgo(static_cast<uint8_t>(track->algo()), static_cast<uint8_t>(track->originalAlgo()));

      const bool use_track_properties = track->pt() > min_track_pt_property;
      if (use_track_properties) {
        packed.setFirstHit(track->hitPattern().getHitPattern(reco::HitPattern::TRACK_HITS, 0));
        if (std::abs(cand->pdgId()) == 22) {
          packed.setTrackProperties(*track, covariancePackingSchemas[4], covarianceVersion);
        } else if (track->hitPattern().numberOfValidPixelHits() > min_valid_pixel_hits) {
          packed.setTrackProperties(*track, covariancePackingSchemas[0], covarianceVersion);
        } else {
          packed.setTrackProperties(*track, covariancePackingSchemas[1], covarianceVersion);
        }
      } else if (packed.pt() > min_track_pt_property) {
        if (track->hitPattern().numberOfValidPixelHits() > 0) {
          packed.setTrackProperties(*track, covariancePackingSchemas[2], covarianceVersion);
        } else {
          packed.setTrackProperties(*track, covariancePackingSchemas[3], covarianceVersion);
        }
      }
      packed.setTrackHighPurity(cand->trackRef().isNonnull() && cand->trackRef()->quality(reco::Track::highPurity));
    } else {
      math::XYZPoint pv_ass_pos = pv_ass->position();
      packed = pat::PackedCandidate(
          cand->polarP4(), pv_ass_pos, cand->pt(), cand->eta(), cand->phi(), cand->pdgId(), pvRefProd, pv_ass.key());
      packed.setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(pat::PackedCandidate::UsedInFitTight));
    }
    return packed;
  }

  const double jet_radius_;
  const double min_candidate_pt_;

  const edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  const edm::EDGetTokenT<VertexCollection> vtx_token_;
  const edm::EDGetTokenT<SVCollection> sv_token_;
  const edm::EDGetTokenT<edm::View<reco::Candidate>> candidateToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> track_builder_token_;

  const double min_jet_pt_;
  const double max_jet_eta_;

  const bool fallback_vertex_association_;

  const edm::EDGetTokenT<edm::Association<VertexCollection>> vertex_associator_token_;
  const edm::EDGetTokenT<edm::ValueMap<int>> vertex_associator_quality_token_;
};

HLTParticleTransformerAK4TagInfoProducer::HLTParticleTransformerAK4TagInfoProducer(const edm::ParameterSet& iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_candidate_pt_(iConfig.getParameter<double>("min_candidate_pt")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      candidateToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("candidates"))),
      track_builder_token_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      max_jet_eta_(iConfig.getParameter<double>("max_jet_eta")),
      fallback_vertex_association_(iConfig.getParameter<bool>("fallback_vertex_association")),
      vertex_associator_token_(
          consumes<edm::Association<VertexCollection>>(iConfig.getParameter<edm::InputTag>("vertex_associator"))),
      vertex_associator_quality_token_(
          consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vertex_associator"))) {
  produces<TagInfoCollection>();
}

void HLTParticleTransformerAK4TagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("jet_radius", 0.4);
  desc.add<double>("min_candidate_pt", 0.95);
  desc.add<edm::InputTag>("vertices", edm::InputTag("hltOfflinePrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("hltInclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("jets", edm::InputTag("hltAK4PFPuppiJets"));
  desc.add<edm::InputTag>("candidates", edm::InputTag("hltParticleFlowTmp"));
  desc.add<double>("min_jet_pt", 15.0);
  desc.add<double>("max_jet_eta", 2.5);
  desc.add<bool>("fallback_vertex_association", false);
  desc.add<edm::InputTag>("vertex_associator", edm::InputTag("hltPrimaryVertexAssociation", "original"));
  descriptions.addWithDefaultLabel(desc);
}

void HLTParticleTransformerAK4TagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output_tag_infos = std::make_unique<TagInfoCollection>();

  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);

  edm::Handle<VertexCollection> vtxs;
  iEvent.getByToken(vtx_token_, vtxs);
  if (vtxs->empty()) {
    iEvent.put(std::move(output_tag_infos));
    return;
  }
  const auto& pv = vtxs->at(0);
  std::unique_ptr<reco::VertexRefProd> PVRefProd = std::make_unique<reco::VertexRefProd>(vtxs);

  edm::Handle<edm::View<reco::Candidate>> tracks;
  iEvent.getByToken(candidateToken_, tracks);

  edm::Handle<SVCollection> svs;
  iEvent.getByToken(sv_token_, svs);

  bool use_vertex_association = !fallback_vertex_association_;
  edm::Handle<edm::ValueMap<int>> pvasq_value_map;
  edm::Handle<edm::Association<VertexCollection>> pvas;
  if (use_vertex_association) {
    iEvent.getByToken(vertex_associator_quality_token_, pvasq_value_map);
    iEvent.getByToken(vertex_associator_token_, pvas);
    if (!pvasq_value_map.isValid() || !pvas.isValid())
      use_vertex_association = false;
  }

  edm::ESHandle<TransientTrackBuilder> track_builder = iSetup.getHandle(track_builder_token_);

  for (std::size_t jet_n = 0; jet_n < jets->size(); ++jet_n) {
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);
    const auto& jet = jets->at(jet_n);

    btagbtvdeep::HLTParticleTransformerAK4Features hltFeatures;
    if (jet.pt() < min_jet_pt_ || std::abs(jet.eta()) > max_jet_eta_) {
      hltFeatures.is_filled = false;
      hltFeatures.global_features.jet_pt = 0.f;
      hltFeatures.global_features.jet_eta = 0.f;
      hltFeatures.global_features.jet_phi = 0.f;
      hltFeatures.global_features.jet_energy = 0.f;
    } else {
      hltFeatures.is_filled = true;

      // Fill secondary vertex features
      {
        SVCollection svs_sorted = *svs;
        std::sort(svs_sorted.begin(), svs_sorted.end(), [&pv](const auto& sv1, const auto& sv2) {
          return btagbtvdeep::sv_vertex_comparator(sv1, sv2, pv);
        });

        GlobalVector jet_vec(jet.px(), jet.py(), jet.pz());
        for (const auto& sv : svs_sorted) {
          if (reco::deltaR2(sv, jet) > (jet_radius_ * jet_radius_))
            continue;

          HLTVtxFeatures svfeat;
          svfeat.jet_sv_pt = sv.pt();
          svfeat.jet_sv_deta = sv.eta() - jet.eta();
          svfeat.jet_sv_dphi = sv.phi() - jet.phi();
          svfeat.jet_sv_eta = sv.eta();
          svfeat.jet_sv_phi = sv.phi();
          svfeat.jet_sv_energy = sv.energy();
          svfeat.jet_sv_mass = sv.mass();
          svfeat.jet_sv_ntrack = sv.numberOfDaughters();
          svfeat.jet_sv_chi2 = sv.vertexNormalizedChi2();

          reco::Vertex::CovarianceMatrix csv;
          sv.fillVertexCovariance(csv);
          reco::Vertex svtx(sv.vertex(), csv);

          VertexDistanceXY dxy;
          auto dxy_meas = dxy.signedDistance(svtx, pv, jet_vec);
          svfeat.jet_sv_dxy = dxy_meas.value();
          svfeat.jet_sv_dxysig = std::fabs(dxy_meas.significance());

          VertexDistance3D d3d;
          auto d3d_meas = d3d.signedDistance(svtx, pv, jet_vec);
          svfeat.jet_sv_d3d = d3d_meas.value();
          svfeat.jet_sv_d3dsig = std::fabs(d3d_meas.significance());
          svfeat.jet_sv_pt_log = std::log(sv.pt());

          const float cos_sv_pv = btagbtvdeep::vertexDdotP(sv, pv);
          svfeat.jet_sv_costhetasvpv = cos_sv_pv;
          svfeat.jet_sv_enratio = (jet.energy() > 0.f ? sv.energy() / jet.energy() : 0.f);

          hltFeatures.vtx_features.push_back(svfeat);
        }
      }

      // Collect and sort PF candidates by pt
      std::vector<const reco::PFCandidate*> pfCandidates;
      for (unsigned int i = 0; i < jet.numberOfDaughters(); ++i) {
        const auto* cand = dynamic_cast<const reco::PFCandidate*>(jet.daughter(i));
        if (!cand || cand->pt() < min_candidate_pt_)
          continue;
        pfCandidates.push_back(cand);
      }

      std::sort(pfCandidates.begin(), pfCandidates.end(), [](const reco::PFCandidate* a, const reco::PFCandidate* b) {
        return a->pt() > b->pt();
      });

      hltFeatures.cpf_candidates.reserve(pfCandidates.size());

      for (const auto* cand : pfCandidates) {
        const reco::Track* track = cand->bestTrack();

        // PV association
        int pv_ass_quality = 0;
        reco::VertexRef pv_ass(vtxs, 0);

        if (use_vertex_association) {
          edm::Ref<edm::View<reco::Candidate>> candRef = getPersistentCandidate(cand, tracks);
          if (candRef.isNonnull() && pvas.isValid() && pvasq_value_map.isValid()) {
            const reco::VertexRef& pv_orig = (*pvas)[candRef];
            if (pv_orig.isNonnull())
              pv_ass = pv_orig;
            pv_ass_quality = (*pvasq_value_map)[candRef];
          }
        }

        // Fallback: find closest PV by dz
        if (!use_vertex_association && track && pv_ass.key() == 0) {
          float z_dist = 99999.f;
          int pv_pos = 0;
          for (size_t iv = 0; iv < vtxs->size(); iv++) {
            float dz = std::abs(track->dz((*vtxs)[iv].position()));
            if (dz < z_dist) {
              z_dist = dz;
              pv_pos = iv;
            }
          }
          pv_ass = reco::VertexRef(vtxs, pv_pos);
        }

        // Compute fromPV inline
        int pvAssocQual;
        if (track) {
          pvAssocQual = static_cast<int>(btagbtvdeep::vtx_ass_from_pfcand(*cand, pv_ass_quality, pv_ass));
        } else {
          // Trackless candidates treated as UsedInFitTight
          pvAssocQual = pat::PackedCandidate::UsedInFitTight;
        }

        // Build PackedCandidate to get packed track properties
        pat::PackedCandidate packed_candidate = buildPackedCandidate(cand, track, pvAssocQual, pv_ass, *PVRefProd);
        math::XYZPoint pv_ass_pos = pv_ass->position();

        const reco::Track* packed_track = packed_candidate.bestTrack();
        bool highPurity = packed_candidate.trackHighPurity();

        // Fill features
        HLTCpfCandidateFeatures feat;
        feat.jet_pfcand_deta = jet.eta() - cand->eta();
        feat.jet_pfcand_dphi = reco::deltaPhi(jet.phi(), cand->phi());
        feat.jet_pfcand_pt_log = (cand->pt() > 0) ? std::log(cand->pt()) : 0;
        feat.jet_pfcand_energy_log = (cand->energy() > 0) ? std::log(cand->energy()) : 0;
        feat.jet_pfcand_charge = static_cast<float>(cand->charge());
        feat.jet_pfcand_frompv = static_cast<float>(packed_candidate.fromPV());
        feat.jet_pfcand_nlostinnerhits = packed_candidate.lostInnerHits();

        if (packed_track) {
          feat.jet_pfcand_track_chi2 = packed_track->normalizedChi2();
          feat.jet_pfcand_track_qual = packed_track->qualityMask();
          feat.jet_pfcand_dz = packed_candidate.dz(pv_ass_pos);
          feat.jet_pfcand_dzsig = std::fabs(packed_candidate.dz(pv_ass_pos) / packed_candidate.dzError());
          feat.jet_pfcand_dxy = packed_candidate.dxy(pv_ass_pos);
          feat.jet_pfcand_dxysig = std::fabs(packed_candidate.dxy(pv_ass_pos) / packed_candidate.dxyError());
          feat.jet_pfcand_npixhits = packed_candidate.numberOfPixelHits();
          feat.jet_pfcand_nstriphits = packed_candidate.stripLayersWithMeasurement();
        } else {
          feat.jet_pfcand_track_chi2 = 0;
          feat.jet_pfcand_track_qual = 0;
          feat.jet_pfcand_dz = 0;
          feat.jet_pfcand_dzsig = 0;
          feat.jet_pfcand_dxy = 0;
          feat.jet_pfcand_dxysig = 0;
          feat.jet_pfcand_npixhits = 0;
          feat.jet_pfcand_nstriphits = 0;
        }

        // etarel from candidate momentum
        feat.jet_pfcand_etarel = reco::btau::etaRel(jet.momentum().Unit(), cand->momentum());

        TVector3 jet_direction(jet.px(), jet.py(), jet.pz());
        jet_direction = jet_direction.Unit();
        TVector3 cand_direction(cand->px(), cand->py(), cand->pz());
        float cand_mag = cand_direction.Mag();
        feat.jet_pfcand_pperp_ratio = (cand_mag > 0) ? jet_direction.Perp(cand_direction) / cand_mag : 0;
        feat.jet_pfcand_ppara_ratio = (cand_mag > 0) ? jet_direction.Dot(cand_direction) / cand_mag : 0;

        // IP tools computed w.r.t. PV[0]
        if (track) {
          reco::TransientTrack tt = track_builder->build(*track);
          GlobalVector jet_global_dir(jet.px(), jet.py(), jet.pz());

          Measurement1D meas_ip3d = IPTools::signedImpactParameter3D(tt, jet_global_dir, pv).second;
          Measurement1D meas_jetdist = IPTools::jetTrackDistance(tt, jet_global_dir, pv).second;
          Measurement1D meas_decayL = IPTools::signedDecayLength3D(tt, jet_global_dir, pv).second;

          feat.jet_pfcand_trackjet_d3d = meas_ip3d.value();
          feat.jet_pfcand_trackjet_d3dsig = std::fabs(meas_ip3d.significance());
          feat.jet_pfcand_trackjet_dist = -meas_jetdist.value();
          feat.jet_pfcand_trackjet_decayL = meas_decayL.value();
        } else {
          feat.jet_pfcand_trackjet_d3d = 0;
          feat.jet_pfcand_trackjet_d3dsig = 0;
          feat.jet_pfcand_trackjet_dist = 0;
          feat.jet_pfcand_trackjet_decayL = 0;
        }
        feat.jet_pfcand_highpurity = highPurity ? 1.f : 0.f;
        feat.jet_pfcand_id = static_cast<float>(std::abs(cand->pdgId()));

        feat.jet_pfcand_pt = cand->pt();
        feat.jet_pfcand_eta = cand->eta();
        feat.jet_pfcand_phi = cand->phi();
        feat.jet_pfcand_energy = cand->energy();

        hltFeatures.cpf_candidates.push_back(feat);
      }

      // Global features
      hltFeatures.global_features.jet_pt = jet.pt();
      hltFeatures.global_features.jet_eta = jet.eta();
      hltFeatures.global_features.jet_phi = jet.phi();
      hltFeatures.global_features.jet_energy = jet.energy();
    }

    output_tag_infos->emplace_back(reco::HLTParticleTransformerAK4TagInfo(hltFeatures, jet_ref));
  }

  iEvent.put(std::move(output_tag_infos));
}

DEFINE_FWK_MODULE(HLTParticleTransformerAK4TagInfoProducer);

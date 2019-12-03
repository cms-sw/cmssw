#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

using namespace btagbtvdeep;

class ParticleNetTagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit ParticleNetTagInfoProducer(const edm::ParameterSet &);
  ~ParticleNetTagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  typedef std::vector<reco::DeepBoostedJetTagInfo> DeepBoostedJetTagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;
  typedef edm::View<reco::Candidate> CandidateView;

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override {}

  void fillParticleFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet);
  void fillSVFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet);

  const double jet_radius_;
  const double min_jet_pt_;
  const double min_pt_for_track_properties_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;
  edm::EDGetTokenT<CandidateView> pfcand_token_;

  bool use_puppi_value_map_;
  bool use_pvasq_value_map_;

  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
  edm::EDGetTokenT<edm::ValueMap<int>> pvasq_value_map_token_;
  edm::EDGetTokenT<edm::Association<VertexCollection>> pvas_token_;

  edm::Handle<VertexCollection> vtxs_;
  edm::Handle<SVCollection> svs_;
  edm::Handle<CandidateView> pfcands_;
  edm::ESHandle<TransientTrackBuilder> track_builder_;
  edm::Handle<edm::ValueMap<float>> puppi_value_map_;
  edm::Handle<edm::ValueMap<int>> pvasq_value_map_;
  edm::Handle<edm::Association<VertexCollection>> pvas_;

  const static std::vector<std::string> particle_features_;
  const static std::vector<std::string> sv_features_;
  const reco::Vertex *pv_ = nullptr;
};

const std::vector<std::string> ParticleNetTagInfoProducer::particle_features_{
    "pfcand_mask",         "pfcand_VTX_ass",        "pfcand_lostInnerHits", "pfcand_quality",
    "pfcand_charge",       "pfcand_isEl",           "pfcand_isMu",          "pfcand_isChargedHad",
    "pfcand_isGamma",      "pfcand_isNeutralHad",   "pfcand_phirel",        "pfcand_etarel",
    "pfcand_abseta",       "pfcand_pt_log_nopuppi", "pfcand_e_log_nopuppi", "pfcand_normchi2",
    "pfcand_dz",           "pfcand_dzsig",          "pfcand_dxy",           "pfcand_dxysig",
    "pfcand_btagEtaRel",   "pfcand_btagPtRatio",    "pfcand_btagPParRatio", "pfcand_btagSip3dVal",
    "pfcand_btagSip3dSig", "pfcand_btagJetDistVal",
};

const std::vector<std::string> ParticleNetTagInfoProducer::sv_features_{
    "sv_mask",
    "sv_phirel",
    "sv_etarel",
    "sv_abseta",
    "sv_mass",
    "sv_pt_log",
    "sv_ntracks",
    "sv_normchi2",
    "sv_dxy",
    "sv_dxysig",
    "sv_d3d",
    "sv_d3dsig",
    "sv_costhetasvpv",
};

ParticleNetTagInfoProducer::ParticleNetTagInfoProducer(const edm::ParameterSet &iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      min_pt_for_track_properties_(iConfig.getParameter<double>("min_pt_for_track_properties")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      pfcand_token_(consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("pf_candidates"))),
      use_puppi_value_map_(false),
      use_pvasq_value_map_(false) {
  const auto &puppi_value_map_tag = iConfig.getParameter<edm::InputTag>("puppi_value_map");
  if (!puppi_value_map_tag.label().empty()) {
    puppi_value_map_token_ = consumes<edm::ValueMap<float>>(puppi_value_map_tag);
    use_puppi_value_map_ = true;
  }

  const auto &pvas_tag = iConfig.getParameter<edm::InputTag>("vertex_associator");
  if (!pvas_tag.label().empty()) {
    pvasq_value_map_token_ = consumes<edm::ValueMap<int>>(pvas_tag);
    pvas_token_ = consumes<edm::Association<VertexCollection>>(pvas_tag);
    use_pvasq_value_map_ = true;
  }

  produces<DeepBoostedJetTagInfoCollection>();
}

ParticleNetTagInfoProducer::~ParticleNetTagInfoProducer() {}

void ParticleNetTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfParticleNetTagInfos
  edm::ParameterSetDescription desc;
  desc.add<double>("jet_radius", 0.8);
  desc.add<double>("min_jet_pt", 150);
  desc.add<double>("min_pt_for_track_properties", -1);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("inclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("pf_candidates", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak8PFJetsPuppi"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("puppi"));
  desc.add<edm::InputTag>("vertex_associator", edm::InputTag("primaryVertexAssociation", "original"));
  descriptions.add("pfParticleNetTagInfos", desc);
}

void ParticleNetTagInfoProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  auto output_tag_infos = std::make_unique<DeepBoostedJetTagInfoCollection>();

  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);

  iEvent.getByToken(vtx_token_, vtxs_);
  if (vtxs_->empty()) {
    // produce empty TagInfos in case no primary vertex
    iEvent.put(std::move(output_tag_infos));
    return;  // exit event
  }
  // primary vertex
  pv_ = &vtxs_->at(0);

  iEvent.getByToken(sv_token_, svs_);

  iEvent.getByToken(pfcand_token_, pfcands_);

  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", track_builder_);

  if (use_puppi_value_map_) {
    iEvent.getByToken(puppi_value_map_token_, puppi_value_map_);
  }

  if (use_pvasq_value_map_) {
    iEvent.getByToken(pvasq_value_map_token_, pvasq_value_map_);
    iEvent.getByToken(pvas_token_, pvas_);
  }

  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++) {
    const auto &jet = (*jets)[jet_n];
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);

    // create jet features
    DeepBoostedJetFeatures features;
    // declare all the feature variables (init as empty vector)
    for (const auto &name : particle_features_) {
      features.add(name);
    }
    for (const auto &name : sv_features_) {
      features.add(name);
    }

    // fill values only if above pt threshold and has daughters, otherwise left
    // empty
    bool fill_vars = true;
    if (jet.pt() < min_jet_pt_)
      fill_vars = false;
    if (jet.numberOfDaughters() == 0)
      fill_vars = false;

    if (fill_vars) {
      fillParticleFeatures(features, jet);
      fillSVFeatures(features, jet);

      features.check_consistency(particle_features_);
      features.check_consistency(sv_features_);
    }

    // this should always be done even if features are not filled
    output_tag_infos->emplace_back(features, jet_ref);
  }

  iEvent.put(std::move(output_tag_infos));
}

void ParticleNetTagInfoProducer::fillParticleFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet) {
  // do nothing if jet does not have constituents
  if (jet.numberOfDaughters() == 0)
    return;

  // some jet properties
  math::XYZVector jet_dir = jet.momentum().Unit();
  GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());
  const float etasign = jet.eta() > 0 ? 1 : -1;

  std::map<reco::CandidatePtr::key_type, float> puppi_wgt_cache;
  auto puppiWgt = [&](const reco::CandidatePtr &cand) {
    const auto *pack_cand = dynamic_cast<const pat::PackedCandidate *>(&(*cand));
    const auto *reco_cand = dynamic_cast<const reco::PFCandidate *>(&(*cand));
    float wgt = 1.;
    if (pack_cand) {
      wgt = pack_cand->puppiWeight();
    } else if (reco_cand) {
      if (use_puppi_value_map_) {
        wgt = (*puppi_value_map_)[cand];
      } else {
        throw edm::Exception(edm::errors::InvalidReference) << "Puppi value map is missing";
      }
    } else {
      throw edm::Exception(edm::errors::InvalidReference) << "Cannot convert to either pat::PackedCandidate or "
                                                             "reco::PFCandidate";
    }
    puppi_wgt_cache[cand.key()] = wgt;
    return wgt;
  };

  std::vector<reco::CandidatePtr> daughters;
  for (const auto &cand : jet.daughterPtrVector()) {
    // remove particles w/ extremely low puppi weights
    if ((puppiWgt(cand)) < 0.01)
      continue;
    // get the original reco/packed candidate not scaled by the puppi weight
    daughters.push_back(pfcands_->ptrAt(cand.key()));
  }
  // sort by original pt (not Puppi-weighted)
  std::sort(daughters.begin(), daughters.end(), [](const auto &a, const auto &b) { return a->pt() > b->pt(); });

  // reserve space
  for (const auto &name : particle_features_) {
    fts.reserve(name, daughters.size());
  }

  auto useTrackProperties = [&](const reco::PFCandidate *reco_cand) {
    const auto *trk = reco_cand->bestTrack();
    return trk != nullptr && trk->pt() > min_pt_for_track_properties_;
  };

  for (const auto &cand : daughters) {
    const auto *packed_cand = dynamic_cast<const pat::PackedCandidate *>(&(*cand));
    const auto *reco_cand = dynamic_cast<const reco::PFCandidate *>(&(*cand));

    if (packed_cand) {
      fts.fill("pfcand_VTX_ass", packed_cand->pvAssociationQuality());
      fts.fill("pfcand_lostInnerHits", packed_cand->lostInnerHits());
      fts.fill("pfcand_quality", packed_cand->bestTrack() ? packed_cand->bestTrack()->qualityMask() : 0);

      fts.fill("pfcand_charge", packed_cand->charge());
      fts.fill("pfcand_isEl", std::abs(packed_cand->pdgId()) == 11);
      fts.fill("pfcand_isMu", std::abs(packed_cand->pdgId()) == 13);
      fts.fill("pfcand_isChargedHad", std::abs(packed_cand->pdgId()) == 211);
      fts.fill("pfcand_isGamma", std::abs(packed_cand->pdgId()) == 22);
      fts.fill("pfcand_isNeutralHad", std::abs(packed_cand->pdgId()) == 130);

      // impact parameters
      fts.fill("pfcand_dz", catch_infs(packed_cand->dz()));
      fts.fill("pfcand_dzsig", packed_cand->bestTrack() ? catch_infs(packed_cand->dz() / packed_cand->dzError()) : 0);
      fts.fill("pfcand_dxy", catch_infs(packed_cand->dxy()));
      fts.fill("pfcand_dxysig",
               packed_cand->bestTrack() ? catch_infs(packed_cand->dxy() / packed_cand->dxyError()) : 0);

    } else if (reco_cand) {
      // get vertex association quality
      int pv_ass_quality = 0;  // fallback value
      float vtx_ass = 0;
      if (use_pvasq_value_map_) {
        pv_ass_quality = (*pvasq_value_map_)[cand];
        const reco::VertexRef &PV_orig = (*pvas_)[cand];
        vtx_ass = vtx_ass_from_pfcand(*reco_cand, pv_ass_quality, PV_orig);
      } else {
        throw edm::Exception(edm::errors::InvalidReference) << "Vertex association missing";
      }

      fts.fill("pfcand_VTX_ass", vtx_ass);
      fts.fill("pfcand_lostInnerHits", useTrackProperties(reco_cand) ? lost_inner_hits_from_pfcand(*reco_cand) : 0);
      fts.fill("pfcand_quality", useTrackProperties(reco_cand) ? quality_from_pfcand(*reco_cand) : 0);

      fts.fill("pfcand_charge", reco_cand->charge());
      fts.fill("pfcand_isEl", std::abs(reco_cand->pdgId()) == 11);
      fts.fill("pfcand_isMu", std::abs(reco_cand->pdgId()) == 13);
      fts.fill("pfcand_isChargedHad", std::abs(reco_cand->pdgId()) == 211);
      fts.fill("pfcand_isGamma", std::abs(reco_cand->pdgId()) == 22);
      fts.fill("pfcand_isNeutralHad", std::abs(reco_cand->pdgId()) == 130);

      // impact parameters
      const auto *trk = reco_cand->bestTrack();
      float dz = trk ? trk->dz(pv_->position()) : 0;
      float dxy = trk ? trk->dxy(pv_->position()) : 0;
      fts.fill("pfcand_dz", catch_infs(dz));
      fts.fill("pfcand_dzsig", trk ? catch_infs(dz / trk->dzError()) : 0);
      fts.fill("pfcand_dxy", catch_infs(dxy));
      fts.fill("pfcand_dxysig", trk ? catch_infs(dxy / trk->dxyError()) : 0);
    }

    // basic kinematics
    fts.fill("pfcand_mask", 1);
    fts.fill("pfcand_phirel", reco::deltaPhi(*cand, jet));
    fts.fill("pfcand_etarel", etasign * (cand->eta() - jet.eta()));
    fts.fill("pfcand_abseta", std::abs(cand->eta()));

    fts.fill("pfcand_pt_log_nopuppi", catch_infs(std::log(cand->pt()), -99));
    fts.fill("pfcand_e_log_nopuppi", catch_infs(std::log(cand->energy()), -99));

    // track info
    const reco::Track *trk = nullptr;
    if (packed_cand) {
      trk = packed_cand->bestTrack();
    } else if (reco_cand && useTrackProperties(reco_cand)) {
      trk = reco_cand->bestTrack();
    }
    if (trk) {
      fts.fill("pfcand_normchi2", catch_infs(std::floor(trk->normalizedChi2())));

      TrackInfoBuilder trkinfo(track_builder_);
      trkinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
      fts.fill("pfcand_btagEtaRel", trkinfo.getTrackEtaRel());
      fts.fill("pfcand_btagPtRatio", trkinfo.getTrackPtRatio());
      fts.fill("pfcand_btagPParRatio", trkinfo.getTrackPParRatio());
      fts.fill("pfcand_btagSip3dVal", trkinfo.getTrackSip3dVal());
      fts.fill("pfcand_btagSip3dSig", trkinfo.getTrackSip3dSig());
      fts.fill("pfcand_btagJetDistVal", trkinfo.getTrackJetDistVal());
    } else {
      fts.fill("pfcand_normchi2", 999);

      fts.fill("pfcand_btagEtaRel", 0);
      fts.fill("pfcand_btagPtRatio", 0);
      fts.fill("pfcand_btagPParRatio", 0);
      fts.fill("pfcand_btagSip3dVal", 0);
      fts.fill("pfcand_btagSip3dSig", 0);
      fts.fill("pfcand_btagJetDistVal", 0);
    }
  }
}

void ParticleNetTagInfoProducer::fillSVFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet) {
  std::vector<const reco::VertexCompositePtrCandidate *> jetSVs;
  for (const auto &sv : *svs_) {
    if (reco::deltaR2(sv, jet) < jet_radius_ * jet_radius_) {
      jetSVs.push_back(&sv);
    }
  }
  // sort by dxy significance
  std::sort(jetSVs.begin(),
            jetSVs.end(),
            [&](const reco::VertexCompositePtrCandidate *sva, const reco::VertexCompositePtrCandidate *svb) {
              return sv_vertex_comparator(*sva, *svb, *pv_);
            });

  // reserve space
  for (const auto &name : sv_features_) {
    fts.reserve(name, jetSVs.size());
  }

  const float etasign = jet.eta() > 0 ? 1 : -1;

  for (const auto *sv : jetSVs) {
    // basic kinematics
    fts.fill("sv_mask", 1);
    fts.fill("sv_phirel", reco::deltaPhi(*sv, jet));
    fts.fill("sv_etarel", etasign * (sv->eta() - jet.eta()));
    fts.fill("sv_abseta", std::abs(sv->eta()));
    fts.fill("sv_mass", sv->mass());
    fts.fill("sv_pt_log", catch_infs(std::log(sv->pt()), -99));

    // sv properties
    fts.fill("sv_ntracks", sv->numberOfDaughters());
    fts.fill("sv_normchi2", catch_infs(sv->vertexNormalizedChi2()));

    const auto &dxy = vertexDxy(*sv, *pv_);
    fts.fill("sv_dxy", dxy.value());
    fts.fill("sv_dxysig", dxy.significance());

    const auto &d3d = vertexD3d(*sv, *pv_);
    fts.fill("sv_d3d", d3d.value());
    fts.fill("sv_d3dsig", d3d.significance());

    fts.fill("sv_costhetasvpv", vertexDdotP(*sv, *pv_));
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(ParticleNetTagInfoProducer);

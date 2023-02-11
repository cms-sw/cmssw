#include <TVector3.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "RecoBTag/FeatureTools/interface/sorting_modules.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

using namespace btagbtvdeep;

class DeepBoostedJetTagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeepBoostedJetTagInfoProducer(const edm::ParameterSet &);
  ~DeepBoostedJetTagInfoProducer() override;

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
  void fillParticleFeaturesHLT(DeepBoostedJetFeatures &fts, const reco::Jet &jet, const reco::VertexRefProd &PVRefProd);
  void fillSVFeaturesHLT(DeepBoostedJetFeatures &fts, const reco::Jet &jet);

  float puppiWgt(const reco::CandidatePtr &cand);
  bool useTrackProperties(const reco::PFCandidate *reco_cand);

  const double jet_radius_;
  const double min_jet_pt_;
  const double max_jet_eta_;
  const double min_pt_for_track_properties_;
  const double min_pt_for_pfcandidates_;
  const bool use_puppiP4_;
  const double min_puppi_wgt_;
  const bool include_neutrals_;
  const bool sort_by_sip2dsig_;
  const bool flip_ip_sign_;
  const double max_sip3dsig_;
  const bool use_hlt_features_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;
  edm::EDGetTokenT<CandidateView> pfcand_token_;

  bool use_puppi_value_map_;
  bool use_pvasq_value_map_;
  bool is_packed_pf_candidate_collection_;

  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
  edm::EDGetTokenT<edm::ValueMap<int>> pvasq_value_map_token_;
  edm::EDGetTokenT<edm::Association<VertexCollection>> pvas_token_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> track_builder_token_;

  edm::Handle<VertexCollection> vtxs_;
  edm::Handle<SVCollection> svs_;
  edm::Handle<CandidateView> pfcands_;
  edm::ESHandle<TransientTrackBuilder> track_builder_;
  edm::Handle<edm::ValueMap<float>> puppi_value_map_;
  edm::Handle<edm::ValueMap<int>> pvasq_value_map_;
  edm::Handle<edm::Association<VertexCollection>> pvas_;

  const static std::vector<std::string> particle_features_;
  const static std::vector<std::string> sv_features_;
  const static std::vector<std::string> particle_features_hlt_;
  const static std::vector<std::string> sv_features_hlt_;
  const reco::Vertex *pv_ = nullptr;
  const static float min_track_pt_property_;
  const static int min_valid_pixel_hits_;

  std::map<reco::CandidatePtr::key_type, float> puppi_wgt_cache;
};

const std::vector<std::string> DeepBoostedJetTagInfoProducer::particle_features_{
    "pfcand_puppiw",        "pfcand_hcalFrac",       "pfcand_VTX_ass",      "pfcand_lostInnerHits",
    "pfcand_quality",       "pfcand_charge",         "pfcand_isEl",         "pfcand_isMu",
    "pfcand_isChargedHad",  "pfcand_isGamma",        "pfcand_isNeutralHad", "pfcand_phirel",
    "pfcand_etarel",        "pfcand_deltaR",         "pfcand_abseta",       "pfcand_ptrel_log",
    "pfcand_erel_log",      "pfcand_pt_log",         "pfcand_drminsv",      "pfcand_drsubjet1",
    "pfcand_drsubjet2",     "pfcand_normchi2",       "pfcand_dz",           "pfcand_dzsig",
    "pfcand_dxy",           "pfcand_dxysig",         "pfcand_dptdpt",       "pfcand_detadeta",
    "pfcand_dphidphi",      "pfcand_dxydxy",         "pfcand_dzdz",         "pfcand_dxydz",
    "pfcand_dphidxy",       "pfcand_dlambdadz",      "pfcand_btagEtaRel",   "pfcand_btagPtRatio",
    "pfcand_btagPParRatio", "pfcand_btagSip2dVal",   "pfcand_btagSip2dSig", "pfcand_btagSip3dVal",
    "pfcand_btagSip3dSig",  "pfcand_btagJetDistVal", "pfcand_mask",         "pfcand_pt_log_nopuppi",
    "pfcand_e_log_nopuppi", "pfcand_ptrel",          "pfcand_erel"};

const std::vector<std::string> DeepBoostedJetTagInfoProducer::particle_features_hlt_{"jet_pfcand_pt_log",
                                                                                     "jet_pfcand_energy_log",
                                                                                     "jet_pfcand_deta",
                                                                                     "jet_pfcand_dphi",
                                                                                     "jet_pfcand_eta",
                                                                                     "jet_pfcand_charge",
                                                                                     "jet_pfcand_frompv",
                                                                                     "jet_pfcand_nlostinnerhits",
                                                                                     "jet_pfcand_track_chi2",
                                                                                     "jet_pfcand_track_qual",
                                                                                     "jet_pfcand_dz",
                                                                                     "jet_pfcand_dzsig",
                                                                                     "jet_pfcand_dxy",
                                                                                     "jet_pfcand_dxysig",
                                                                                     "jet_pfcand_etarel",
                                                                                     "jet_pfcand_pperp_ratio",
                                                                                     "jet_pfcand_ppara_ratio",
                                                                                     "jet_pfcand_trackjet_d3d",
                                                                                     "jet_pfcand_trackjet_d3dsig",
                                                                                     "jet_pfcand_trackjet_dist",
                                                                                     "jet_pfcand_nhits",
                                                                                     "jet_pfcand_npixhits",
                                                                                     "jet_pfcand_nstriphits",
                                                                                     "jet_pfcand_trackjet_decayL",
                                                                                     "jet_pfcand_puppiw",
                                                                                     "pfcand_mask"};

const std::vector<std::string> DeepBoostedJetTagInfoProducer::sv_features_{"sv_mask",
                                                                           "sv_ptrel",
                                                                           "sv_erel",
                                                                           "sv_phirel",
                                                                           "sv_etarel",
                                                                           "sv_deltaR",
                                                                           "sv_abseta",
                                                                           "sv_mass",
                                                                           "sv_ptrel_log",
                                                                           "sv_erel_log",
                                                                           "sv_pt_log",
                                                                           "sv_pt",
                                                                           "sv_ntracks",
                                                                           "sv_normchi2",
                                                                           "sv_dxy",
                                                                           "sv_dxysig",
                                                                           "sv_d3d",
                                                                           "sv_d3dsig",
                                                                           "sv_costhetasvpv"};

const std::vector<std::string> DeepBoostedJetTagInfoProducer::sv_features_hlt_{"jet_sv_pt_log",
                                                                               "jet_sv_mass",
                                                                               "jet_sv_deta",
                                                                               "jet_sv_dphi",
                                                                               "jet_sv_eta",
                                                                               "jet_sv_ntrack",
                                                                               "jet_sv_chi2",
                                                                               "jet_sv_dxy",
                                                                               "jet_sv_dxysig",
                                                                               "jet_sv_d3d",
                                                                               "jet_sv_d3dsig",
                                                                               "sv_mask"};

const float DeepBoostedJetTagInfoProducer::min_track_pt_property_ = 0.5;
const int DeepBoostedJetTagInfoProducer::min_valid_pixel_hits_ = 0;

DeepBoostedJetTagInfoProducer::DeepBoostedJetTagInfoProducer(const edm::ParameterSet &iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      max_jet_eta_(iConfig.getParameter<double>("max_jet_eta")),
      min_pt_for_track_properties_(iConfig.getParameter<double>("min_pt_for_track_properties")),
      min_pt_for_pfcandidates_(iConfig.getParameter<double>("min_pt_for_pfcandidates")),
      use_puppiP4_(iConfig.getParameter<bool>("use_puppiP4")),
      min_puppi_wgt_(iConfig.getParameter<double>("min_puppi_wgt")),
      include_neutrals_(iConfig.getParameter<bool>("include_neutrals")),
      sort_by_sip2dsig_(iConfig.getParameter<bool>("sort_by_sip2dsig")),
      flip_ip_sign_(iConfig.getParameter<bool>("flip_ip_sign")),
      max_sip3dsig_(iConfig.getParameter<double>("sip3dSigMax")),
      use_hlt_features_(iConfig.getParameter<bool>("use_hlt_features")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      pfcand_token_(consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("pf_candidates"))),
      use_puppi_value_map_(false),
      use_pvasq_value_map_(false),
      track_builder_token_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))) {
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

DeepBoostedJetTagInfoProducer::~DeepBoostedJetTagInfoProducer() {}

void DeepBoostedJetTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfDeepBoostedJetTagInfos
  edm::ParameterSetDescription desc;
  desc.add<double>("jet_radius", 0.8);
  desc.add<double>("min_jet_pt", 150);
  desc.add<double>("max_jet_eta", 99);
  desc.add<double>("min_pt_for_track_properties", -1);
  desc.add<double>("min_pt_for_pfcandidates", -1);
  desc.add<bool>("use_puppiP4", true);
  desc.add<bool>("include_neutrals", true);
  desc.add<bool>("sort_by_sip2dsig", false);
  desc.add<double>("min_puppi_wgt", 0.01);
  desc.add<bool>("flip_ip_sign", false);
  desc.add<double>("sip3dSigMax", -1);
  desc.add<bool>("use_hlt_features", false);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("inclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("pf_candidates", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak8PFJetsPuppi"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("puppi"));
  desc.add<edm::InputTag>("vertex_associator", edm::InputTag("primaryVertexAssociation", "original"));
  descriptions.add("pfDeepBoostedJetTagInfos", desc);
}

void DeepBoostedJetTagInfoProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // output collection
  auto output_tag_infos = std::make_unique<DeepBoostedJetTagInfoCollection>();
  // Input jets
  auto jets = iEvent.getHandle(jet_token_);
  // Primary vertexes
  iEvent.getByToken(vtx_token_, vtxs_);
  if (vtxs_->empty()) {
    // produce empty TagInfos in case no primary vertex
    iEvent.put(std::move(output_tag_infos));
    return;  // exit event
  }
  // Leading vertex
  pv_ = &vtxs_->at(0);
  // Secondary vertexs
  iEvent.getByToken(sv_token_, svs_);
  // PF candidates
  iEvent.getByToken(pfcand_token_, pfcands_);
  is_packed_pf_candidate_collection_ = false;
  if (dynamic_cast<const pat::PackedCandidateCollection *>(pfcands_.product()))
    is_packed_pf_candidate_collection_ = true;
  // Track builder
  track_builder_ = iSetup.getHandle(track_builder_token_);
  // Puppi weight value map
  if (use_puppi_value_map_)
    iEvent.getByToken(puppi_value_map_token_, puppi_value_map_);
  // Vertex associator map
  if (use_pvasq_value_map_) {
    iEvent.getByToken(pvasq_value_map_token_, pvasq_value_map_);
    iEvent.getByToken(pvas_token_, pvas_);
  }

  // Loop over jet
  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++) {
    const auto &jet = (*jets)[jet_n];
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);

    // create jet features
    DeepBoostedJetFeatures features;
    if (not use_hlt_features_) {
      for (const auto &name : particle_features_) {
        features.add(name);
      }
      for (const auto &name : sv_features_) {
        features.add(name);
      }
    } else {
      // declare all the feature variables (init as empty vector)
      for (const auto &name : particle_features_hlt_) {
        features.add(name);
      }
      for (const auto &name : sv_features_hlt_) {
        features.add(name);
      }
    }

    // fill values only if above pt threshold and has daughters, otherwise left
    bool fill_vars = true;
    if (jet.pt() < min_jet_pt_ or std::abs(jet.eta()) > max_jet_eta_)
      fill_vars = false;
    if (jet.numberOfDaughters() == 0)
      fill_vars = false;

    // fill features
    if (fill_vars) {
      fillParticleFeatures(features, jet);
      fillSVFeatures(features, jet);
      if (use_hlt_features_) {
        features.check_consistency(particle_features_hlt_);
        features.check_consistency(sv_features_hlt_);
      } else {
        features.check_consistency(particle_features_);
        features.check_consistency(sv_features_);
      }
    }
    // this should always be done even if features are not filled
    output_tag_infos->emplace_back(features, jet_ref);
  }
  // move output collection
  iEvent.put(std::move(output_tag_infos));
}

float DeepBoostedJetTagInfoProducer::puppiWgt(const reco::CandidatePtr &cand) {
  const auto *pack_cand = dynamic_cast<const pat::PackedCandidate *>(&(*cand));
  const auto *reco_cand = dynamic_cast<const reco::PFCandidate *>(&(*cand));
  float wgt = 1.;
  if (pack_cand){
    //fallback value
    wgt = pack_cand->puppiWeight();
    if (use_puppi_value_map_)
      wgt = (*puppi_value_map_)[cand];
  }
  else if (reco_cand) {
    if (use_puppi_value_map_)
      wgt = (*puppi_value_map_)[cand];
  } else
    throw edm::Exception(edm::errors::InvalidReference)
        << "Cannot convert to either pat::PackedCandidate or reco::PFCandidate";
  puppi_wgt_cache[cand.key()] = wgt;
  return wgt;
}

bool DeepBoostedJetTagInfoProducer::useTrackProperties(const reco::PFCandidate *reco_cand) {
  const auto *track = reco_cand->bestTrack();
  return track != nullptr and track->pt() > min_pt_for_track_properties_;
};

void DeepBoostedJetTagInfoProducer::fillParticleFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet) {
  // some jet properties
  math::XYZVector jet_dir = jet.momentum().Unit();
  TVector3 jet_direction(jet.momentum().Unit().x(), jet.momentum().Unit().y(), jet.momentum().Unit().z());
  GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());
  const float etasign = jet.eta() > 0 ? 1 : -1;
  // vertexes
  reco::VertexRefProd PVRefProd(vtxs_);
  // track builder
  TrackInfoBuilder trackinfo(track_builder_);

  // make list of pf-candidates to be considered
  std::vector<reco::CandidatePtr> daughters;
  for (const auto &dau : jet.daughterPtrVector()) {
    // remove particles w/ extremely low puppi weights
    // [Note] use jet daughters here to get the puppiWgt correctly
    if ((puppiWgt(dau)) < min_puppi_wgt_)
      continue;
    // from here: get the original reco/packed candidate not scaled by the puppi weight
    auto cand = pfcands_->ptrAt(dau.key());
    // base requirements on PF candidates
    if (use_hlt_features_ and cand->pt() < min_pt_for_pfcandidates_)
      continue;
    // charged candidate selection (for Higgs Interaction Net)
    if (!include_neutrals_ and (cand->charge() == 0 or cand->pt() < min_pt_for_track_properties_))
      continue;
    // only when computing the nagative tagger: remove charged candidates with high sip3d
    if (flip_ip_sign_ and cand->charge()) {
      trackinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
      if (trackinfo.getTrackSip3dSig() > max_sip3dsig_)
        continue;
    }
    // filling daughters
    daughters.push_back(cand);
  }

  // Sorting of PF-candidates
  std::vector<btagbtvdeep::SortingClass<reco::CandidatePtr>> c_sorted;
  if (sort_by_sip2dsig_) {
    // sort charged pf candidates by 2d impact parameter significance
    for (const auto &cand : daughters) {
      trackinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
      c_sorted.emplace_back(cand,
                            trackinfo.getTrackSip2dSig(),
                            -btagbtvdeep::mindrsvpfcand(*svs_, &(*cand), jet_radius_),
                            cand->pt() / jet.pt());
    }
    std::sort(c_sorted.begin(), c_sorted.end(), btagbtvdeep::SortingClass<reco::CandidatePtr>::compareByABCInv);
    for (unsigned int i = 0; i < c_sorted.size(); i++) {
      const auto &c = c_sorted.at(i);
      const auto &cand = c.get();
      daughters.at(i) = cand;
    }
  } else {
    // sort by Puppi-weighted pt
    if (use_puppiP4_)
      std::sort(daughters.begin(), daughters.end(), [&](const reco::CandidatePtr &a, const reco::CandidatePtr &b) {
        return puppi_wgt_cache.at(a.key()) * a->pt() > puppi_wgt_cache.at(b.key()) * b->pt();
      });
    // sort by original pt (not Puppi-weighted)
    else
      std::sort(daughters.begin(), daughters.end(), [](const auto &a, const auto &b) { return a->pt() > b->pt(); });
  }

  // reserve space
  if (use_hlt_features_) {
    for (const auto &name : particle_features_hlt_)
      fts.reserve(name, daughters.size());
  } else {
    for (const auto &name : particle_features_)
      fts.reserve(name, daughters.size());
  }

  // build white list of candidates i.e. particles and tracks belonging to SV. Needed when input particles are not packed PF candidates
  std::vector<unsigned int> whiteListSV;
  std::vector<reco::TrackRef> whiteListTk;
  if (not is_packed_pf_candidate_collection_) {
    for (size_t isv = 0; isv < svs_->size(); isv++) {
      for (size_t icand = 0; icand < svs_->at(isv).numberOfSourceCandidatePtrs(); icand++) {
        const edm::Ptr<reco::Candidate> &cand = svs_->at(isv).sourceCandidatePtr(icand);
        if (cand.id() == pfcands_.id())
          whiteListSV.push_back(cand.key());
      }
      for (auto cand = svs_->at(isv).begin(); cand != svs_->at(isv).end(); cand++) {
        const reco::RecoChargedCandidate *chCand = dynamic_cast<const reco::RecoChargedCandidate *>(&(*cand));
        if (chCand != nullptr) {
          whiteListTk.push_back(chCand->track());
        }
      }
    }
  }

  // Build observables
  size_t icand = 0;
  for (const auto &cand : daughters) {
    const auto *packed_cand = dynamic_cast<const pat::PackedCandidate *>(&(*cand));
    const auto *reco_cand = dynamic_cast<const reco::PFCandidate *>(&(*cand));

    if (not packed_cand and not reco_cand)
      throw edm::Exception(edm::errors::InvalidReference)
          << "Cannot convert to either reco::PFCandidate or pat::PackedCandidate";

    if (!include_neutrals_ and
        ((packed_cand and !packed_cand->hasTrackDetails()) or (reco_cand and !useTrackProperties(reco_cand)))) {
      icand++;
      continue;
    }

    const float ip_sign = flip_ip_sign_ ? -1 : 1;

    // input particle is a packed PF candidate
    auto candP4 = use_puppiP4_ ? puppi_wgt_cache.at(cand.key()) * cand->p4() : cand->p4();
    auto candP3 = use_puppiP4_ ? puppi_wgt_cache.at(cand.key()) * cand->momentum() : cand->momentum();

    // candidate track
    const reco::Track *track = nullptr;
    if (packed_cand)
      track = packed_cand->bestTrack();
    else if (reco_cand and useTrackProperties(reco_cand))
      track = reco_cand->bestTrack();

    // reco-vertex association
    int pv_ass_quality = 0;
    reco::VertexRef pv_ass;
    float vtx_ass = 0;
    if (reco_cand) {
      if (use_pvasq_value_map_) {
        pv_ass_quality = (*pvasq_value_map_)[cand];
        pv_ass = (*pvas_)[cand];
        vtx_ass = vtx_ass_from_pfcand(*reco_cand, pv_ass_quality, pv_ass);
      } else
        throw edm::Exception(edm::errors::InvalidReference) << "Vertex association missing";
    }

    // Building offline features
    if (not use_hlt_features_) {
      // in case of packed candidate
      if (packed_cand) {
        float hcal_fraction = 0.;
        if (packed_cand->pdgId() == 1 or packed_cand->pdgId() == 130)
          hcal_fraction = packed_cand->hcalFraction();
        else if (packed_cand->isIsolatedChargedHadron())
          hcal_fraction = packed_cand->rawHcalFraction();

        fts.fill("pfcand_hcalFrac", hcal_fraction);
        fts.fill("pfcand_VTX_ass", packed_cand->pvAssociationQuality());
        fts.fill("pfcand_lostInnerHits", packed_cand->lostInnerHits());
        fts.fill("pfcand_quality", track ? track->qualityMask() : 0);
        fts.fill("pfcand_charge", packed_cand->charge());
        fts.fill("pfcand_isEl", std::abs(packed_cand->pdgId()) == 11);
        fts.fill("pfcand_isMu", std::abs(packed_cand->pdgId()) == 13);
        fts.fill("pfcand_isChargedHad", std::abs(packed_cand->pdgId()) == 211);
        fts.fill("pfcand_isGamma", std::abs(packed_cand->pdgId()) == 22);
        fts.fill("pfcand_isNeutralHad", std::abs(packed_cand->pdgId()) == 130);
        fts.fill("pfcand_dz", ip_sign * packed_cand->dz());
        fts.fill("pfcand_dxy", ip_sign * packed_cand->dxy());
        fts.fill("pfcand_dzsig", track ? ip_sign * packed_cand->dz() / packed_cand->dzError() : 0);
        fts.fill("pfcand_dxysig", track ? ip_sign * packed_cand->dxy() / packed_cand->dxyError() : 0);

      }
      // in the case of reco candidate
      else if (reco_cand) {
        fts.fill("pfcand_hcalFrac", reco_cand->hcalEnergy() / (reco_cand->ecalEnergy() + reco_cand->hcalEnergy()));
        fts.fill("pfcand_VTX_ass", vtx_ass);
        fts.fill("pfcand_lostInnerHits", useTrackProperties(reco_cand) ? lost_inner_hits_from_pfcand(*reco_cand) : 0);
        fts.fill("pfcand_quality", useTrackProperties(reco_cand) ? quality_from_pfcand(*reco_cand) : 0);
        fts.fill("pfcand_charge", reco_cand->charge());
        fts.fill("pfcand_isEl", std::abs(reco_cand->pdgId()) == 11);
        fts.fill("pfcand_isMu", std::abs(reco_cand->pdgId()) == 13);
        fts.fill("pfcand_isChargedHad", std::abs(reco_cand->pdgId()) == 211);
        fts.fill("pfcand_isGamma", std::abs(reco_cand->pdgId()) == 22);
        fts.fill("pfcand_isNeutralHad", std::abs(reco_cand->pdgId()) == 130);
        fts.fill("pfcand_dz", track ? ip_sign * track->dz(pv_->position()) : 0);
        fts.fill("pfcand_dzsig", track ? ip_sign * track->dz(pv_->position()) / track->dzError() : 0);
        fts.fill("pfcand_dxy", track ? ip_sign * track->dxy(pv_->position()) : 0);
        fts.fill("pfcand_dxysig", track ? ip_sign * track->dxy(pv_->position()) / track->dxyError() : 0);
      }

      // generic candidate observables
      fts.fill("pfcand_puppiw", puppi_wgt_cache.at(cand.key()));
      fts.fill("pfcand_phirel", reco::deltaPhi(candP4, jet));
      fts.fill("pfcand_etarel", etasign * (candP4.eta() - jet.eta()));
      fts.fill("pfcand_deltaR", reco::deltaR(candP4, jet));
      fts.fill("pfcand_abseta", std::abs(candP4.eta()));

      fts.fill("pfcand_ptrel_log", std::log(candP4.pt() / jet.pt()));
      fts.fill("pfcand_ptrel", candP4.pt() / jet.pt());
      fts.fill("pfcand_erel_log", std::log(candP4.energy() / jet.energy()));
      fts.fill("pfcand_erel", candP4.energy() / jet.energy());
      fts.fill("pfcand_pt_log", std::log(candP4.pt()));

      fts.fill("pfcand_mask", 1);
      fts.fill("pfcand_pt_log_nopuppi", std::log(cand->pt()));
      fts.fill("pfcand_e_log_nopuppi", std::log(cand->energy()));

      float drminpfcandsv = btagbtvdeep::mindrsvpfcand(*svs_, &(*cand), std::numeric_limits<float>::infinity());
      fts.fill("pfcand_drminsv", drminpfcandsv);

      if (track) {
        auto cov = [&](unsigned i, unsigned j) { return track->covariance(i, j); };
        fts.fill("pfcand_dptdpt", cov(0, 0));
        fts.fill("pfcand_detadeta", cov(1, 1));
        fts.fill("pfcand_dphidphi", cov(2, 2));
        fts.fill("pfcand_dxydxy", cov(3, 3));
        fts.fill("pfcand_dzdz", cov(4, 4));
        fts.fill("pfcand_dxydz", cov(3, 4));
        fts.fill("pfcand_dphidxy", cov(2, 3));
        fts.fill("pfcand_dlambdadz", cov(1, 4));

        fts.fill("pfcand_normchi2", std::floor(track->normalizedChi2()));

        trackinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
        fts.fill("pfcand_btagEtaRel", trackinfo.getTrackEtaRel());
        fts.fill("pfcand_btagPtRatio", trackinfo.getTrackPtRatio());
        fts.fill("pfcand_btagPParRatio", trackinfo.getTrackPParRatio());
        fts.fill("pfcand_btagSip2dVal", ip_sign * trackinfo.getTrackSip2dVal());
        fts.fill("pfcand_btagSip2dSig", ip_sign * trackinfo.getTrackSip2dSig());
        fts.fill("pfcand_btagSip3dVal", ip_sign * trackinfo.getTrackSip3dVal());
        fts.fill("pfcand_btagSip3dSig", ip_sign * trackinfo.getTrackSip3dSig());
        fts.fill("pfcand_btagJetDistVal", trackinfo.getTrackJetDistVal());
      } else {
        fts.fill("pfcand_normchi2", 999);
        fts.fill("pfcand_dptdpt", 0);
        fts.fill("pfcand_detadeta", 0);
        fts.fill("pfcand_dphidphi", 0);
        fts.fill("pfcand_dxydxy", 0);
        fts.fill("pfcand_dzdz", 0);
        fts.fill("pfcand_dxydz", 0);
        fts.fill("pfcand_dphidxy", 0);
        fts.fill("pfcand_dlambdadz", 0);
        fts.fill("pfcand_btagEtaRel", 0);
        fts.fill("pfcand_btagPtRatio", 0);
        fts.fill("pfcand_btagPParRatio", 0);
        fts.fill("pfcand_btagSip2dVal", 0);
        fts.fill("pfcand_btagSip2dSig", 0);
        fts.fill("pfcand_btagSip3dVal", 0);
        fts.fill("pfcand_btagSip3dSig", 0);
        fts.fill("pfcand_btagJetDistVal", 0);
      }

      // subjets only if the incomming jets is a PAT one
      const auto *patJet = dynamic_cast<const pat::Jet *>(&jet);
      if (patJet and patJet->nSubjetCollections() > 0) {
        auto subjets = patJet->subjets();
        std::sort(subjets.begin(), subjets.end(), [](const edm::Ptr<pat::Jet> &p1, const edm::Ptr<pat::Jet> &p2) {
          return p1->pt() > p2->pt();
        });
        fts.fill("pfcand_drsubjet1", !subjets.empty() ? reco::deltaR(*cand, *subjets.at(0)) : -1);
        fts.fill("pfcand_drsubjet2", subjets.size() > 1 ? reco::deltaR(*cand, *subjets.at(1)) : -1);
      } else {
        fts.fill("pfcand_drsubjet1", -1);
        fts.fill("pfcand_drsubjet2", -1);
      }
    }
    // using HLT features
    else {
      pat::PackedCandidate candidate;
      math::XYZPoint pv_ass_pos;
      // In case input is a packed candidate (evaluate HLT network on offline)
      if (packed_cand) {
        candidate = *packed_cand;
        pv_ass = reco::VertexRef(vtxs_, 0);
        pv_ass_pos = pv_ass->position();
      }
      // In case input is a reco::PFCandidate
      else if (reco_cand) {
        // follow what is done in PhysicsTools/PatAlgos/plugins/PATPackedCandidateProducer.cc to minimize differences between HLT and RECO choices of input observables
        // no reference to vertex, take closest in dz or position 0
        if (not pv_ass.isNonnull()) {
          if (track) {
            float z_dist = 99999;
            int pv_pos = -1;
            for (size_t iv = 0; iv < vtxs_->size(); iv++) {
              float dz = std::abs(track->dz(((*vtxs_)[iv]).position()));
              if (dz < z_dist) {
                z_dist = dz;
                pv_pos = iv;
              }
            }
            pv_ass = reco::VertexRef(vtxs_, pv_pos);
          } else
            pv_ass = reco::VertexRef(vtxs_, 0);
        }
        pv_ass_pos = pv_ass->position();

        // create a transient packed candidate
        if (track) {
          candidate = pat::PackedCandidate(cand->polarP4(),
                                           track->referencePoint(),
                                           track->pt(),
                                           track->eta(),
                                           track->phi(),
                                           cand->pdgId(),
                                           PVRefProd,
                                           pv_ass.key());
          candidate.setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(
              btagbtvdeep::vtx_ass_from_pfcand(*reco_cand, pv_ass_quality, pv_ass)));
          candidate.setCovarianceVersion(0);
          pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
          int nlost = track->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
          if (nlost == 0) {
            if (track->hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1))
              lostHits = pat::PackedCandidate::validHitInFirstPixelBarrelLayer;
          } else
            lostHits = (nlost == 1 ? pat::PackedCandidate::oneLostInnerHit : pat::PackedCandidate::moreLostInnerHits);
          candidate.setLostInnerHits(lostHits);

          if (useTrackProperties(reco_cand) or
              std::find(whiteListSV.begin(), whiteListSV.end(), icand) != whiteListSV.end() or
              std::find(whiteListTk.begin(), whiteListTk.end(), reco_cand->trackRef()) != whiteListTk.end()) {
            candidate.setFirstHit(track->hitPattern().getHitPattern(reco::HitPattern::TRACK_HITS, 0));
            if (abs(cand->pdgId()) == 22)
              candidate.setTrackProperties(*track, 0, 0);
            else {
              if (track->hitPattern().numberOfValidPixelHits() > min_valid_pixel_hits_)
                candidate.setTrackProperties(*track, 8, 0);
              else
                candidate.setTrackProperties(*track, 264, 0);
            }
          } else {
            if (candidate.pt() > min_track_pt_property_) {
              if (track->hitPattern().numberOfValidPixelHits() > 0)
                candidate.setTrackProperties(*track, 520, 0);
              else
                candidate.setTrackProperties(*track, 776, 0);
            }
          }
          candidate.setTrackHighPurity(reco_cand->trackRef().isNonnull() and
                                       reco_cand->trackRef()->quality(reco::Track::highPurity));
        } else {
          candidate = pat::PackedCandidate(
              cand->polarP4(), pv_ass_pos, cand->pt(), cand->eta(), cand->phi(), cand->pdgId(), PVRefProd, pv_ass.key());
          candidate.setAssociationQuality(
              pat::PackedCandidate::PVAssociationQuality(pat::PackedCandidate::UsedInFitTight));
        }
        /// override track
        track = candidate.bestTrack();
      }

      TVector3 cand_direction(candP3.x(), candP3.y(), candP3.z());

      fts.fill("jet_pfcand_pt_log", std::log(candP4.pt()));
      fts.fill("jet_pfcand_energy_log", std::log(candP4.energy()));
      fts.fill("jet_pfcand_eta", candP4.eta());
      fts.fill("jet_pfcand_deta", jet_direction.Eta() - cand_direction.Eta());
      fts.fill("jet_pfcand_dphi", jet_direction.DeltaPhi(cand_direction));
      fts.fill("jet_pfcand_charge", cand->charge());
      fts.fill("jet_pfcand_etarel", reco::btau::etaRel(jet_dir, candP3));
      fts.fill("jet_pfcand_pperp_ratio", jet_direction.Perp(cand_direction) / cand_direction.Mag());
      fts.fill("jet_pfcand_ppara_ratio", jet_direction.Dot(cand_direction) / cand_direction.Mag());
      fts.fill("jet_pfcand_frompv", candidate.fromPV());
      fts.fill("jet_pfcand_dz", candidate.dz(pv_ass_pos));
      fts.fill("jet_pfcand_dxy", candidate.dxy(pv_ass_pos));
      fts.fill("jet_pfcand_puppiw", puppi_wgt_cache.at(cand.key()));
      fts.fill("jet_pfcand_nlostinnerhits", candidate.lostInnerHits());
      fts.fill("jet_pfcand_nhits", candidate.numberOfHits());
      fts.fill("jet_pfcand_npixhits", candidate.numberOfPixelHits());
      fts.fill("jet_pfcand_nstriphits", candidate.stripLayersWithMeasurement());
      fts.fill("pfcand_mask", 1);

      if (track) {
        fts.fill("jet_pfcand_dzsig", fabs(candidate.dz(pv_ass_pos)) / candidate.dzError());
        fts.fill("jet_pfcand_dxysig", fabs(candidate.dxy(pv_ass_pos)) / candidate.dxyError());
        fts.fill("jet_pfcand_track_chi2", track->normalizedChi2());
        fts.fill("jet_pfcand_track_qual", track->qualityMask());

        reco::TransientTrack transientTrack = track_builder_->build(*track);
        Measurement1D meas_ip2d =
            IPTools::signedTransverseImpactParameter(transientTrack, jet_ref_track_dir, *pv_).second;
        Measurement1D meas_ip3d = IPTools::signedImpactParameter3D(transientTrack, jet_ref_track_dir, *pv_).second;
        Measurement1D meas_jetdist = IPTools::jetTrackDistance(transientTrack, jet_ref_track_dir, *pv_).second;
        Measurement1D meas_decayl = IPTools::signedDecayLength3D(transientTrack, jet_ref_track_dir, *pv_).second;

        fts.fill("jet_pfcand_trackjet_d3d", meas_ip3d.value());
        fts.fill("jet_pfcand_trackjet_d3dsig", fabs(meas_ip3d.significance()));
        fts.fill("jet_pfcand_trackjet_dist", -meas_jetdist.value());
        fts.fill("jet_pfcand_trackjet_decayL", meas_decayl.value());
      } else {
        fts.fill("jet_pfcand_dzsig", 0);
        fts.fill("jet_pfcand_dxysig", 0);
        fts.fill("jet_pfcand_track_chi2", 0);
        fts.fill("jet_pfcand_track_qual", 0);
        fts.fill("jet_pfcand_trackjet_d3d", 0);
        fts.fill("jet_pfcand_trackjet_d3dsig", 0);
        fts.fill("jet_pfcand_trackjet_dist", 0);
        fts.fill("jet_pfcand_trackjet_decayL", 0);
      }
    }
    icand++;
  }
}

void DeepBoostedJetTagInfoProducer::fillSVFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet) {
  // secondary vertexes matching jet
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
  if (not use_hlt_features_) {
    for (const auto &name : sv_features_) {
      fts.reserve(name, jetSVs.size());
    }
  } else {
    for (const auto &name : sv_features_hlt_) {
      fts.reserve(name, jetSVs.size());
    }
  }

  const float etasign = jet.eta() > 0 ? 1 : -1;

  GlobalVector jet_global_vec(jet.px(), jet.py(), jet.pz());

  for (const auto *sv : jetSVs) {
    // features for reco
    if (not use_hlt_features_) {
      fts.fill("sv_mask", 1);
      fts.fill("sv_phirel", reco::deltaPhi(*sv, jet));
      fts.fill("sv_etarel", etasign * (sv->eta() - jet.eta()));
      fts.fill("sv_deltaR", reco::deltaR(*sv, jet));
      fts.fill("sv_abseta", std::abs(sv->eta()));
      fts.fill("sv_mass", sv->mass());

      fts.fill("sv_ptrel_log", std::log(sv->pt() / jet.pt()));
      fts.fill("sv_ptrel", sv->pt() / jet.pt());
      fts.fill("sv_erel_log", std::log(sv->energy() / jet.energy()));
      fts.fill("sv_erel", sv->energy() / jet.energy());
      fts.fill("sv_pt_log", std::log(sv->pt()));
      fts.fill("sv_pt", sv->pt());

      fts.fill("sv_ntracks", sv->numberOfDaughters());
      fts.fill("sv_normchi2", sv->vertexNormalizedChi2());
      const auto &dxy = vertexDxy(*sv, *pv_);
      fts.fill("sv_dxy", dxy.value());
      fts.fill("sv_dxysig", dxy.significance());
      const auto &d3d = vertexD3d(*sv, *pv_);
      fts.fill("sv_d3d", d3d.value());
      fts.fill("sv_d3dsig", d3d.significance());

      fts.fill("sv_costhetasvpv", (flip_ip_sign_ ? -1.f : 1.f) * vertexDdotP(*sv, *pv_));
    } else {
      fts.fill("sv_mask", 1);
      fts.fill("jet_sv_pt_log", log(sv->pt()));
      fts.fill("jet_sv_eta", sv->eta());
      fts.fill("jet_sv_mass", sv->mass());
      fts.fill("jet_sv_deta", sv->eta() - jet.eta());
      fts.fill("jet_sv_dphi", sv->phi() - jet.phi());
      fts.fill("jet_sv_ntrack", sv->numberOfDaughters());
      fts.fill("jet_sv_chi2", sv->vertexNormalizedChi2());

      reco::Vertex::CovarianceMatrix csv;
      sv->fillVertexCovariance(csv);
      reco::Vertex svtx(sv->vertex(), csv);

      VertexDistanceXY dxy;
      auto valxy = dxy.signedDistance(svtx, *pv_, jet_global_vec);
      fts.fill("jet_sv_dxy", valxy.value());
      fts.fill("jet_sv_dxysig", fabs(valxy.significance()));

      VertexDistance3D d3d;
      auto val3d = d3d.signedDistance(svtx, *pv_, jet_global_vec);
      fts.fill("jet_sv_d3d", val3d.value());
      fts.fill("jet_sv_d3dsig", fabs(val3d.significance()));
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(DeepBoostedJetTagInfoProducer);

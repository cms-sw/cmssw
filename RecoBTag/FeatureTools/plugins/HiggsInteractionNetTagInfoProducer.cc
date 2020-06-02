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
#include "RecoBTag/FeatureTools/interface/sorting_modules.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/HiggsInteractionNetFeatures.h"
#include "DataFormats/BTauReco/interface/HiggsInteractionNetTagInfo.h"

using namespace btagbtvdeep;

class HiggsInteractionNetTagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit HiggsInteractionNetTagInfoProducer(const edm::ParameterSet &);
  ~HiggsInteractionNetTagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  typedef std::vector<reco::HiggsInteractionNetTagInfo> HiggsInteractionNetTagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;
  typedef edm::View<reco::Candidate> CandidateView;

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endStream() override {}

  void fillChargedParticleFeatures(HiggsInteractionNetFeatures &fts, const reco::Jet &jet);
  void fillSVFeatures(HiggsInteractionNetFeatures &fts, const reco::Jet &jet);

  const double jet_radius_;
  const double min_jet_pt_;
  const double min_pt_for_track_properties_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;
  edm::EDGetTokenT<CandidateView> pfcand_token_;

  edm::Handle<VertexCollection> vtxs_;
  edm::Handle<SVCollection> svs_;
  edm::Handle<CandidateView> pfcands_;
  edm::ESHandle<TransientTrackBuilder> track_builder_;

  const static std::vector<std::string> cpf_features_;
  const static std::vector<std::string> sv_features_;
  const reco::Vertex *pv_ = nullptr;
};

const std::vector<std::string> HiggsInteractionNetTagInfoProducer::cpf_features_{
    "cpf_ptrel",        "cpf_erel",         "cpf_phirel",       "cpf_etarel",       "cpf_deltaR",
    "cpf_drminsv",      "cpf_drsubjet1",    "cpf_drsubjet2",    "cpf_dz",           "cpf_dzsig",
    "cpf_dxy",          "cpf_dxysig",       "cpf_normchi2",     "cpf_quality",      "cpf_dptdpt",
    "cpf_detadeta",     "cpf_dphidphi",     "cpf_dxydxy",       "cpf_dzdz",         "cpf_dxydz",
    "cpf_dphidxy",      "cpf_dlambdadz",    "cpf_btagEtaRel",   "cpf_btagPtRatio",  "cpf_btagPParRatio",
    "cpf_btagSip2dVal", "cpf_btagSip2dSig", "cpf_btagSip3dVal", "cpf_btagSip3dSig", "cpf_btagJetDistVal"};

const std::vector<std::string> HiggsInteractionNetTagInfoProducer::sv_features_{"sv_ptrel",
                                                                                "sv_erel",
                                                                                "sv_phirel",
                                                                                "sv_etarel",
                                                                                "sv_deltaR",
                                                                                "sv_pt",
                                                                                "sv_mass",
                                                                                "sv_ntracks",
                                                                                "sv_normchi2",
                                                                                "sv_costhetasvpv",
                                                                                "sv_dxy",
                                                                                "sv_dxysig",
                                                                                "sv_d3d",
                                                                                "sv_d3dsig"};

HiggsInteractionNetTagInfoProducer::HiggsInteractionNetTagInfoProducer(const edm::ParameterSet &iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      min_pt_for_track_properties_(iConfig.getParameter<double>("min_pt_for_track_properties")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      pfcand_token_(consumes<CandidateView>(iConfig.getParameter<edm::InputTag>("pf_candidates"))) {
  produces<HiggsInteractionNetTagInfoCollection>();
}

HiggsInteractionNetTagInfoProducer::~HiggsInteractionNetTagInfoProducer() {}

void HiggsInteractionNetTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  // pfHiggsInteractionNetTagInfos
  edm::ParameterSetDescription desc;
  desc.add<double>("jet_radius", 0.8);
  desc.add<double>("min_jet_pt", 150);
  desc.add<double>("min_pt_for_track_properties", 0.95);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("inclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("pf_candidates", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak8PFJetsPuppi"));
  descriptions.add("pfHiggsInteractionNetTagInfos", desc);
}

void HiggsInteractionNetTagInfoProducer::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  auto output_tag_infos = std::make_unique<HiggsInteractionNetTagInfoCollection>();

  auto jets = iEvent.getHandle(jet_token_);

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

  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++) {
    const auto &jet = (*jets)[jet_n];
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);

    // create jet features
    HiggsInteractionNetFeatures features;
    // declare all the feature variables (init as empty vector)
    for (const auto &name : cpf_features_) {
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
      fillChargedParticleFeatures(features, jet);
      fillSVFeatures(features, jet);

      features.check_consistency(cpf_features_);
      features.check_consistency(sv_features_);
    }

    // this should always be done even if features are not filled
    output_tag_infos->emplace_back(features, jet_ref);
  }

  iEvent.put(std::move(output_tag_infos));
}

void HiggsInteractionNetTagInfoProducer::fillChargedParticleFeatures(HiggsInteractionNetFeatures &fts,
                                                                     const reco::Jet &jet) {
  // require the input to be a pat::Jet
  const auto *patJet = dynamic_cast<const pat::Jet *>(&jet);
  if (!patJet) {
    throw edm::Exception(edm::errors::InvalidReference) << "Input is not a pat::Jet.";
  }

  // do nothing if jet does not have constituents
  if (jet.numberOfDaughters() == 0)
    return;

  // some jet properties
  math::XYZVector jet_dir = jet.momentum().Unit();
  GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());
  const float etasign = jet.eta() > 0 ? 1 : -1;

  std::vector<reco::CandidatePtr> daughters;
  for (const auto &cand : jet.daughterPtrVector()) {
    // get the original reco/packed candidate
    auto daugh = pfcands_->ptrAt(cand.key());
    if (daugh->charge() != 0 && daugh->pt() > min_pt_for_track_properties_)
      daughters.push_back(daugh);
  }

  // reserve space
  for (const auto &name : cpf_features_) {
    fts.reserve(name, daughters.size());
  }

  auto useTrackProperties = [&](const reco::PFCandidate *reco_cand) {
    const auto *trk = reco_cand->bestTrack();
    return trk != nullptr && trk->pt() > min_pt_for_track_properties_;
  };

  // sort charged pf candidates by 2d impact parameter significance
  std::vector<btagbtvdeep::SortingClass<reco::CandidatePtr>> c_sorted;
  for (const auto &cand : daughters) {
    TrackInfoBuilder trkinfo(track_builder_);
    trkinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
    c_sorted.emplace_back(cand,
                          trkinfo.getTrackSip2dSig(),
                          -btagbtvdeep::mindrsvpfcand(*svs_, &(*cand), jet_radius_),
                          cand->pt() / jet.pt());
  }

  std::sort(c_sorted.begin(), c_sorted.end(), btagbtvdeep::SortingClass<reco::CandidatePtr>::compareByABCInv);

  for (const auto c : c_sorted) {
    const auto &cand = c.get();
    //for (const auto &cand : daughters) {
    const auto *packed_cand = dynamic_cast<const pat::PackedCandidate *>(&(*cand));
    const auto *reco_cand = dynamic_cast<const reco::PFCandidate *>(&(*cand));

    if (packed_cand && !packed_cand->hasTrackDetails()) {
      continue;
    } else if (reco_cand && !useTrackProperties(reco_cand)) {
      continue;
    }

    auto candP4 = cand->p4();
    if (packed_cand) {
      fts.fill("cpf_quality", packed_cand->pseudoTrack().qualityMask());

      // impact parameters
      fts.fill("cpf_dz", catch_infs(packed_cand->dz()));
      fts.fill("cpf_dzsig", catch_infs_and_bound(packed_cand->dz() / packed_cand->dzError(), 0, -2000, 2000));
      fts.fill("cpf_dxy", catch_infs(packed_cand->dxy()));
      fts.fill("cpf_dxysig", catch_infs_and_bound(packed_cand->dxy() / packed_cand->dxyError(), 0, -2000, 2000));
    } else if (reco_cand) {
      fts.fill("cpf_quality", quality_from_pfcand(*reco_cand));

      // impact parameters
      const auto *trk = reco_cand->bestTrack();
      float dz = trk->dz(pv_->position());
      float dxy = trk->dxy(pv_->position());
      fts.fill("cpf_dz", catch_infs(dz));
      fts.fill("cpf_dzsig", catch_infs_and_bound(dz / trk->dzError(), 0, -2000, 2000));
      fts.fill("cpf_dxy", catch_infs(dxy));
      fts.fill("cpf_dxysig", catch_infs_and_bound(dxy / trk->dxyError(), 0, -2000, 2000));
    }

    // basic kinematics
    fts.fill("cpf_phirel", reco::deltaPhi(candP4, jet));
    fts.fill("cpf_etarel", etasign * (candP4.eta() - jet.eta()));
    fts.fill("cpf_deltaR", reco::deltaR(candP4, jet));

    fts.fill("cpf_ptrel", catch_infs(candP4.pt() / jet.pt(), -99));
    fts.fill("cpf_erel", catch_infs(candP4.energy() / jet.energy(), -99));

    float drminpfcandsv = btagbtvdeep::mindrsvpfcand(*svs_, &(*cand), jet_radius_);
    fts.fill("cpf_drminsv", catch_infs_and_bound(drminpfcandsv, 0, -1. * jet_radius_, 0, -1. * jet_radius_));

    // subjets
    auto subjets = patJet->subjets();
    std::sort(subjets.begin(), subjets.end(), [](const edm::Ptr<pat::Jet> &p1, const edm::Ptr<pat::Jet> &p2) {
      return p1->pt() > p2->pt();
    });  // sort by pt
    fts.fill("cpf_drsubjet1", !subjets.empty() ? reco::deltaR(*cand, *subjets.at(0)) : -1);
    fts.fill("cpf_drsubjet2", subjets.size() > 1 ? reco::deltaR(*cand, *subjets.at(1)) : -1);

    const reco::Track *trk = nullptr;
    if (packed_cand && packed_cand->hasTrackDetails()) {
      trk = &(packed_cand->pseudoTrack());
    } else if (reco_cand && useTrackProperties(reco_cand)) {
      trk = reco_cand->bestTrack();
    }
    TrackInfoBuilder trkinfo(track_builder_);
    trkinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
    fts.fill("cpf_btagEtaRel", catch_infs_and_bound(trkinfo.getTrackEtaRel(), 0, -5, 15));
    fts.fill("cpf_btagPtRatio", catch_infs_and_bound(trkinfo.getTrackPtRatio(), 0, -1, 10));
    fts.fill("cpf_btagPParRatio", catch_infs_and_bound(trkinfo.getTrackPParRatio(), 0, -10, 100));
    fts.fill("cpf_btagSip2dVal", catch_infs_and_bound(trkinfo.getTrackSip2dVal(), 0, -1, 70));
    fts.fill("cpf_btagSip2dSig", catch_infs_and_bound(trkinfo.getTrackSip2dSig(), 0, -1, 4e4));
    fts.fill("cpf_btagSip3dVal", catch_infs_and_bound(trkinfo.getTrackSip3dVal(), 0, -1, 1e5));
    fts.fill("cpf_btagSip3dSig", catch_infs_and_bound(trkinfo.getTrackSip3dSig(), 0, -1, 4e4));
    fts.fill("cpf_btagJetDistVal", catch_infs_and_bound(trkinfo.getTrackJetDistVal(), 0, -20, 1));
    fts.fill("cpf_normchi2", catch_infs_and_bound(std::floor(trk->normalizedChi2()), 300, -1, 300));

    // track covariance
    auto cov = [&](unsigned i, unsigned j) { return catch_infs(trk->covariance(i, j)); };
    fts.fill("cpf_dptdpt", cov(0, 0));
    fts.fill("cpf_detadeta", cov(1, 1));
    fts.fill("cpf_dphidphi", cov(2, 2));
    fts.fill("cpf_dxydxy", cov(3, 3));
    fts.fill("cpf_dzdz", cov(4, 4));
    fts.fill("cpf_dxydz", cov(3, 4));
    fts.fill("cpf_dphidxy", cov(2, 3));
    fts.fill("cpf_dlambdadz", cov(1, 4));
  }
}

void HiggsInteractionNetTagInfoProducer::fillSVFeatures(HiggsInteractionNetFeatures &fts, const reco::Jet &jet) {
  std::vector<const reco::VertexCompositePtrCandidate *> jetSVs;
  for (const auto &sv : *svs_) {
    if (reco::deltaR2(sv, jet) < jet_radius_ * jet_radius_) {
      jetSVs.push_back(&sv);
    }
  }
  // sort secondary vertices by dxy significance
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
    fts.fill("sv_phirel", reco::deltaPhi(*sv, jet));
    fts.fill("sv_etarel", etasign * (sv->eta() - jet.eta()));
    fts.fill("sv_deltaR", catch_infs_and_bound(std::fabs(reco::deltaR(*sv, jet)) - 0.5, 0, -2, 0));
    fts.fill("sv_mass", sv->mass());

    fts.fill("sv_ptrel", catch_infs(sv->pt() / jet.pt(), -99));
    fts.fill("sv_erel", catch_infs(sv->energy() / jet.energy(), -99));
    fts.fill("sv_pt", catch_infs(sv->pt(), -99));

    // sv properties
    fts.fill("sv_ntracks", sv->numberOfDaughters());
    fts.fill("sv_normchi2", catch_infs_and_bound(sv->vertexNormalizedChi2(), 1000, -1000, 1000));

    const auto &dxy = vertexDxy(*sv, *pv_);
    fts.fill("sv_dxy", catch_infs(dxy.value()));
    fts.fill("sv_dxysig", catch_infs_and_bound(dxy.significance(), 0, -1, 800));

    const auto &d3d = vertexD3d(*sv, *pv_);
    fts.fill("sv_d3d", catch_infs(d3d.value()));
    fts.fill("sv_d3dsig", catch_infs_and_bound(d3d.significance(), 0, -1, 800));

    fts.fill("sv_costhetasvpv", vertexDdotP(*sv, *pv_));
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HiggsInteractionNetTagInfoProducer);

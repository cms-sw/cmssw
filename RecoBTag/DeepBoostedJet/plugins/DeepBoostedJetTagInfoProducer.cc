#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoBTag/TensorFlow/interface/TrackInfoBuilder.h"
#include "RecoBTag/TensorFlow/interface/deep_helpers.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/BTauReco/interface/DeepBoostedJetTagInfo.h"

using namespace btagbtvdeep;

class DeepBoostedJetTagInfoProducer : public edm::stream::EDProducer<>
{

public:
  explicit DeepBoostedJetTagInfoProducer(const edm::ParameterSet&);
  ~DeepBoostedJetTagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::vector<reco::DeepBoostedJetTagInfo> DeepBoostedJetTagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override{}

  void fillParticleFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet);
  void fillSVFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet);

  const bool update_jets_;
  const double jet_radius_;
  const double min_jet_pt_;
  const double min_pt_for_track_properties_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<edm::View<reco::Jet>> sdjet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;

  bool use_puppi_value_map_;
  bool use_pvasq_value_map_;
  bool use_subjet_collection_;

  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
  edm::EDGetTokenT<edm::ValueMap<int>> pvasq_value_map_token_;
  edm::EDGetTokenT<edm::Association<VertexCollection>> pvas_token_;

  edm::Handle<edm::View<reco::Jet>> sdjets;
  edm::Handle<VertexCollection> vtxs;
  edm::Handle<SVCollection> svs;
  edm::ESHandle<TransientTrackBuilder> track_builder;
  edm::Handle<edm::ValueMap<float>> puppi_value_map;
  edm::Handle<edm::ValueMap<int>> pvasq_value_map;
  edm::Handle<edm::Association<VertexCollection>> pvas;

  std::vector<std::string> feature_names_;
  const reco::Vertex *pv_ = nullptr;
};

DeepBoostedJetTagInfoProducer::DeepBoostedJetTagInfoProducer(const edm::ParameterSet& iConfig)
: update_jets_(iConfig.getParameter<bool>("updateJetCollection"))
, jet_radius_(iConfig.getParameter<double>("jet_radius"))
, min_jet_pt_(iConfig.getParameter<double>("min_jet_pt"))
, min_pt_for_track_properties_(iConfig.getParameter<double>("minPtForTrackProperties"))
, jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets")))
, vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices")))
, sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices")))
, use_puppi_value_map_(false)
, use_pvasq_value_map_(false)
, use_subjet_collection_(false)
, feature_names_(iConfig.getParameter<std::vector<std::string>>("feature_names"))
{

  const auto & puppi_value_map_tag = iConfig.getParameter<edm::InputTag>("puppi_value_map");
  if (!puppi_value_map_tag.label().empty()) {
    puppi_value_map_token_ = consumes<edm::ValueMap<float>>(puppi_value_map_tag);
    use_puppi_value_map_ = true;
  }

  const auto & pvas_tag = iConfig.getParameter<edm::InputTag>("vertex_associator");
  if (!pvas_tag.label().empty()) {
    pvasq_value_map_token_ = consumes<edm::ValueMap<int>>(pvas_tag);
    pvas_token_ = consumes<edm::Association<VertexCollection>>(pvas_tag);
    use_pvasq_value_map_ = true;
  }

  const auto & subjet_tag = iConfig.getParameter<edm::InputTag>("subjets");
  if (!subjet_tag.label().empty()) {
    sdjet_token_ = consumes<edm::View<reco::Jet>>(subjet_tag);
    use_subjet_collection_ = true;
  }

  produces<DeepBoostedJetTagInfoCollection>();

}

DeepBoostedJetTagInfoProducer::~DeepBoostedJetTagInfoProducer()
{
}

void DeepBoostedJetTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  // pfDeepBoostedJetTagInfos
  edm::ParameterSetDescription desc;
  // updateJetCollection
  // set to true if applying on existing jet collections (whose daughters are *not* puppi weighted)
  // set to false if the jet collection is (re)clustered (whose daughters are puppi weighted)
  desc.add<bool>("updateJetCollection", true);
  desc.add<double>("jet_radius", 0.8);
  desc.add<double>("min_jet_pt", 170);
  desc.add<double>("minPtForTrackProperties", -1);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("inclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("jets", edm::InputTag("slimmedJetsAK8"));
  desc.add<edm::InputTag>("subjets", edm::InputTag("slimmedJetsAK8PFPuppiSoftDropPacked"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("puppi"));
  desc.add<edm::InputTag>("vertex_associator", edm::InputTag("primaryVertexAssociation","original"));
  desc.add<std::vector<std::string>>("feature_names", std::vector<std::string>{
    "pfcand_puppiw",
    "pfcand_hcalFrac",
    "pfcand_VTX_ass",
    "pfcand_lostInnerHits",
    "pfcand_quality",
    "pfcand_charge",
    "pfcand_isEl",
    "pfcand_isMu",
    "pfcand_isChargedHad",
    "pfcand_isGamma",
    "pfcand_isNeutralHad",
    "pfcand_pt",
    "pfcand_ptrel",
    "pfcand_erel",
    "pfcand_phirel",
    "pfcand_etarel",
    "pfcand_deltaR",
    "pfcand_abseta",
    "pfcand_ptrel_log",
    "pfcand_erel_log",
    "pfcand_pt_log",
    "pfcand_drminsv",
    "pfcand_drsubjet1",
    "pfcand_drsubjet2",
    "pfcand_normchi2",
    "pfcand_dz",
    "pfcand_dzsig",
    "pfcand_dxy",
    "pfcand_dxysig",
    "pfcand_dptdpt",
    "pfcand_detadeta",
    "pfcand_dphidphi",
    "pfcand_dxydxy",
    "pfcand_dzdz",
    "pfcand_dxydz",
    "pfcand_dphidxy",
    "pfcand_dlambdadz",
    "pfcand_btagMomentum",
    "pfcand_btagEta",
    "pfcand_btagEtaRel",
    "pfcand_btagPtRel",
    "pfcand_btagPPar",
    "pfcand_btagDeltaR",
    "pfcand_btagPtRatio",
    "pfcand_btagPParRatio",
    "pfcand_btagSip2dVal",
    "pfcand_btagSip2dSig",
    "pfcand_btagSip3dVal",
    "pfcand_btagSip3dSig",
    "pfcand_btagJetDistVal",
    "sv_ptrel",
    "sv_erel",
    "sv_phirel",
    "sv_etarel",
    "sv_deltaR",
    "sv_pt",
    "sv_abseta",
    "sv_mass",
    "sv_ptrel_log",
    "sv_erel_log",
    "sv_pt_log",
    "sv_ntracks",
    "sv_chi2",
    "sv_ndf",
    "sv_normchi2",
    "sv_dxy",
    "sv_dxyerr",
    "sv_dxysig",
    "sv_d3d",
    "sv_d3derr",
    "sv_d3dsig",
    "sv_costhetasvpv",
  });
  descriptions.add("pfDeepBoostedJetTagInfos", desc);
}

void DeepBoostedJetTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  auto output_tag_infos = std::make_unique<DeepBoostedJetTagInfoCollection>();

  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);

  iEvent.getByToken(vtx_token_, vtxs);
  if (vtxs->empty()){
    // produce empty TagInfos in case no primary vertex
    iEvent.put(std::move(output_tag_infos));
    return; // exit event
  }
  // primary vertex
  pv_ = &vtxs->at(0);

  iEvent.getByToken(sv_token_, svs);

  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", track_builder);

  if (use_puppi_value_map_) {
    iEvent.getByToken(puppi_value_map_token_, puppi_value_map);
  }

  if (use_pvasq_value_map_) {
    iEvent.getByToken(pvasq_value_map_token_, pvasq_value_map);
    iEvent.getByToken(pvas_token_, pvas);
  }

  if (use_subjet_collection_) {
    iEvent.getByToken(sdjet_token_, sdjets);
  }

  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++){

    // reco jet reference (use as much as possible)
    const auto& jet = jets->at(jet_n);
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);

    // create jet features
    DeepBoostedJetFeatures features;
    // declare all the feature variables (init as empty vector)
    for (const auto &name : feature_names_) features.add(name);
    // fill only if above pt threshold
    if (jet.pt() > min_jet_pt_){
      fillParticleFeatures(features, jet);
      fillSVFeatures(features, jet);
    }

    output_tag_infos->emplace_back(features, jet_ref);
  }

  iEvent.put(std::move(output_tag_infos));
}

void DeepBoostedJetTagInfoProducer::fillParticleFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet){

  // stuff required for dealing with pf candidates
  math::XYZVector jet_dir = jet.momentum().Unit();
  GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());
  const float etasign = jet.eta()>0 ? 1 : -1;

  auto puppiWgt = [&](reco::CandidatePtr cand){
    float puppiw = 1; // fall back value
    const edm::Ptr<pat::PackedCandidate> packed_cand(cand);
    if (packed_cand.isNonnull()) { puppiw = packed_cand->puppiWeight(); }
    else{
      if (use_puppi_value_map_){ puppiw = (*puppi_value_map)[cand]; }
      else { throw edm::Exception(edm::errors::InvalidReference) << "PUPPI value map is missing"; }
    }
    return puppiw;
  };

  std::vector<reco::CandidatePtr> daughters;
  for (auto cand : jet.daughterPtrVector()){
    // remove particles w/ extremely low puppi weights
    if ((puppiWgt(cand)) < 0.01) continue;
    daughters.push_back(cand);
  }
  // sort by (Puppi-weighted) pt
  if (update_jets_) {
    // updating jet collection:
    // linked daughters here are the original PackedCandidates
    // need to scale the p4 with their puppi weights
    std::sort(daughters.begin(), daughters.end(), [&](const reco::CandidatePtr a, const reco::CandidatePtr b){ return puppiWgt(a)*a->pt() > puppiWgt(b)*b->pt(); });
  }else{
    std::sort(daughters.begin(), daughters.end(), [](const reco::CandidatePtr a, const reco::CandidatePtr b){ return a->pt() > b->pt(); });
  }

  auto useTrackProperties = [&](const edm::Ptr<reco::PFCandidate> reco_cand) {
    if (reco_cand.isNull()) return false;
    const auto* trk = reco_cand->bestTrack();
    return trk!=nullptr && trk->pt()>min_pt_for_track_properties_;
  };

  for (const auto cand : daughters){
    const edm::Ptr<pat::PackedCandidate> packed_cand(cand);
    const edm::Ptr<reco::PFCandidate> reco_cand(cand);

    auto puppiP4 = cand->p4();
    if (packed_cand.isNonnull()){
      if (update_jets_) {
        // updating jet collection:
        // linked daughters here are the original PackedCandidates
        // need to scale the p4 with their puppi weights
        puppiP4 *= packed_cand->puppiWeight();
      }

      fts.fill("pfcand_puppiw", packed_cand->puppiWeight());
      fts.fill("pfcand_hcalFrac", packed_cand->hcalFraction());
      fts.fill("pfcand_VTX_ass", packed_cand->pvAssociationQuality());
      fts.fill("pfcand_lostInnerHits", packed_cand->lostInnerHits());
      fts.fill("pfcand_quality", packed_cand->bestTrack() ? packed_cand->bestTrack()->qualityMask() : 0);

      fts.fill("pfcand_charge", packed_cand->charge());
      fts.fill("pfcand_isEl", std::abs(packed_cand->pdgId())==11);
      fts.fill("pfcand_isMu", std::abs(packed_cand->pdgId())==13);
      fts.fill("pfcand_isChargedHad", std::abs(packed_cand->pdgId())==211);
      fts.fill("pfcand_isGamma", std::abs(packed_cand->pdgId())==22);
      fts.fill("pfcand_isNeutralHad", std::abs(packed_cand->pdgId())==130);

      // impact parameters
      fts.fill("pfcand_dz", catch_infs(packed_cand->dz()));
      fts.fill("pfcand_dzsig", packed_cand->bestTrack() ? catch_infs(packed_cand->dz()/packed_cand->dzError()) : 0);
      fts.fill("pfcand_dxy", catch_infs(packed_cand->dxy()));
      fts.fill("pfcand_dxysig", packed_cand->bestTrack() ? catch_infs(packed_cand->dxy()/packed_cand->dxyError()) : 0);

    } else if(reco_cand.isNonnull()) {
      // get vertex association quality
      int pv_ass_quality = 0; // fallback value
      float vtx_ass = 0;
      if (use_pvasq_value_map_) {
        pv_ass_quality = (*pvasq_value_map)[reco_cand];
        const reco::VertexRef & PV_orig = (*pvas)[reco_cand];
        vtx_ass = vtx_ass_from_pfcand(*reco_cand, pv_ass_quality, PV_orig);
      } else {
        throw edm::Exception(edm::errors::InvalidReference) << "Vertex association missing";
      }

      fts.fill("pfcand_puppiw", puppiWgt(cand));
      fts.fill("pfcand_hcalFrac", reco_cand->hcalEnergy()/(reco_cand->ecalEnergy()+reco_cand->hcalEnergy()));
      fts.fill("pfcand_VTX_ass", vtx_ass);
      fts.fill("pfcand_lostInnerHits", useTrackProperties(reco_cand) ? lost_inner_hits_from_pfcand(*reco_cand) : 0);
      fts.fill("pfcand_quality", useTrackProperties(reco_cand) ? quality_from_pfcand(*reco_cand) : 0);

      fts.fill("pfcand_charge", reco_cand->charge());
      fts.fill("pfcand_isEl", std::abs(reco_cand->pdgId())==11);
      fts.fill("pfcand_isMu", std::abs(reco_cand->pdgId())==13);
      fts.fill("pfcand_isChargedHad", std::abs(reco_cand->pdgId())==211);
      fts.fill("pfcand_isGamma", std::abs(reco_cand->pdgId())==22);
      fts.fill("pfcand_isNeutralHad", std::abs(reco_cand->pdgId())==130);

      // impact parameters
      const auto* trk = reco_cand->bestTrack();
      float dz = trk ? trk->dz(pv_->position()) : 0;
      float dxy = trk ? trk->dxy(pv_->position()) : 0;
      fts.fill("pfcand_dz", catch_infs(dz));
      fts.fill("pfcand_dzsig", trk ? catch_infs(dz/trk->dzError()) : 0);
      fts.fill("pfcand_dxy", catch_infs(dxy));
      fts.fill("pfcand_dxysig", trk ? catch_infs(dxy/trk->dxyError()) : 0);

    }else {
      throw edm::Exception(edm::errors::InvalidReference) << "Cannot convert reco::Candidate to either pat::PackedCandidate or reco::PFCandidate";
    }

    // basic kinematics, valid for both charged and neutral
    fts.fill("pfcand_pt", puppiP4.pt());
    fts.fill("pfcand_ptrel", puppiP4.pt()/jet.pt());
    fts.fill("pfcand_erel", puppiP4.energy()/jet.energy());
    fts.fill("pfcand_phirel", reco::deltaPhi(puppiP4, jet));
    fts.fill("pfcand_etarel", etasign * (puppiP4.eta() - jet.eta()));
    fts.fill("pfcand_deltaR", reco::deltaR(puppiP4, jet));
    fts.fill("pfcand_abseta", std::abs(puppiP4.eta()));

    fts.fill("pfcand_ptrel_log", catch_infs(std::log(puppiP4.pt()/jet.pt()), -99));
    fts.fill("pfcand_erel_log", catch_infs(std::log(puppiP4.energy()/jet.energy()), -99));
    fts.fill("pfcand_pt_log", catch_infs(std::log(puppiP4.pt()), -99));

    double minDR = 999;
    for (const auto &sv : *svs){
      double dr = reco::deltaR(*cand, sv);
      if (dr < minDR) minDR = dr;
    }
    fts.fill("pfcand_drminsv", minDR==999 ? -1 : minDR);

    std::vector<edm::Ptr<reco::Jet>> subjets;
    if (use_subjet_collection_){
      for (const auto &sj : *sdjets) {
        // sdjets stores the soft-drop AK8 jets, with the actual subjets stored as daughters
        // PhysicsTools/PatAlgos/python/slimming/applySubstructure_cff.py
        // PhysicsTools/PatUtils/plugins/JetSubstructurePacker.cc
        if (reco::deltaR(sj, jet) < jet_radius_) {
          for ( size_t ida = 0; ida < sj.numberOfDaughters(); ++ida ) {
            auto candPtr =  sj.daughterPtr(ida);
            subjets.emplace_back(candPtr);
          }
          break;
        }
      }
    }else {
      try {
        const auto &patJet = dynamic_cast<const pat::Jet&>(jet);
        for (auto sj : patJet.subjets()) { subjets.emplace_back(sj); }
      }catch (const std::bad_cast &e) {
        throw edm::Exception(edm::errors::InvalidReference) << "Cannot access subjets because this is not a pat::Jet.";
      }
    }

    fts.fill("pfcand_drsubjet1", subjets.size()>0 ? reco::deltaR(*cand, *subjets.at(0)) : -1);
    fts.fill("pfcand_drsubjet2", subjets.size()>1 ? reco::deltaR(*cand, *subjets.at(1)) : -1);

    const reco::Track *trk = nullptr;
    if (packed_cand.isNonnull()) { trk = packed_cand->bestTrack(); }
    else if (useTrackProperties(reco_cand)) { trk= reco_cand->bestTrack(); }
    if (trk){
      fts.fill("pfcand_normchi2", catch_infs(std::floor(trk->normalizedChi2())));

      // track covariance
      auto cov = [&](unsigned i, unsigned j) {
        return catch_infs(trk->covariance(i, j));
      };
      fts.fill("pfcand_dptdpt", cov(0,0));
      fts.fill("pfcand_detadeta", cov(1,1));
      fts.fill("pfcand_dphidphi", cov(2,2));
      fts.fill("pfcand_dxydxy", cov(3,3));
      fts.fill("pfcand_dzdz", cov(4,4));
      fts.fill("pfcand_dxydz", cov(3,4));
      fts.fill("pfcand_dphidxy", cov(2,3));
      fts.fill("pfcand_dlambdadz", cov(1,4));

      TrackInfoBuilder trkinfo(track_builder);
      trkinfo.buildTrackInfo(&(*cand), jet_dir, jet_ref_track_dir, *pv_);
      fts.fill("pfcand_btagMomentum", trkinfo.getTrackMomentum());
      fts.fill("pfcand_btagEta", trkinfo.getTrackEta());
      fts.fill("pfcand_btagEtaRel", trkinfo.getTrackEtaRel());
      fts.fill("pfcand_btagPtRel", trkinfo.getTrackPtRel());
      fts.fill("pfcand_btagPPar", trkinfo.getTrackPPar());
      fts.fill("pfcand_btagDeltaR", trkinfo.getTrackDeltaR());
      fts.fill("pfcand_btagPtRatio", trkinfo.getTrackPtRatio());
      fts.fill("pfcand_btagPParRatio", trkinfo.getTrackPParRatio());
      fts.fill("pfcand_btagSip2dVal", trkinfo.getTrackSip2dVal());
      fts.fill("pfcand_btagSip2dSig", trkinfo.getTrackSip2dSig());
      fts.fill("pfcand_btagSip3dVal", trkinfo.getTrackSip3dVal());
      fts.fill("pfcand_btagSip3dSig", trkinfo.getTrackSip3dSig());
      fts.fill("pfcand_btagJetDistVal", trkinfo.getTrackJetDistVal());
    }else{
      fts.fill("pfcand_normchi2", 999);

      fts.fill("pfcand_dptdpt", 0);
      fts.fill("pfcand_detadeta", 0);
      fts.fill("pfcand_dphidphi", 0);
      fts.fill("pfcand_dxydxy", 0);
      fts.fill("pfcand_dzdz", 0);
      fts.fill("pfcand_dxydz", 0);
      fts.fill("pfcand_dphidxy", 0);
      fts.fill("pfcand_dlambdadz", 0);

      fts.fill("pfcand_btagMomentum", 0);
      fts.fill("pfcand_btagEta", 0);
      fts.fill("pfcand_btagEtaRel", 0);
      fts.fill("pfcand_btagPtRel", 0);
      fts.fill("pfcand_btagPPar", 0);
      fts.fill("pfcand_btagDeltaR", 0);
      fts.fill("pfcand_btagPtRatio", 0);
      fts.fill("pfcand_btagPParRatio", 0);
      fts.fill("pfcand_btagSip2dVal", 0);
      fts.fill("pfcand_btagSip2dSig", 0);
      fts.fill("pfcand_btagSip3dVal", 0);
      fts.fill("pfcand_btagSip3dSig", 0);
      fts.fill("pfcand_btagJetDistVal", 0);
    }

  }

}

void DeepBoostedJetTagInfoProducer::fillSVFeatures(DeepBoostedJetFeatures &fts, const reco::Jet &jet){
  std::vector<const reco::VertexCompositePtrCandidate*> jetSVs;
  for (const auto &sv : *svs){
    if (reco::deltaR(sv, jet) < jet_radius_) {
      jetSVs.push_back(&sv);
    }
  }
  // sort by dxy significance
  std::sort(jetSVs.begin(), jetSVs.end(), [&](const reco::VertexCompositePtrCandidate *sva, const reco::VertexCompositePtrCandidate *svb){
    return sv_vertex_comparator(*sva, *svb, *pv_);
  });

  const float etasign = jet.eta()>0 ? 1 : -1;

  for (const auto *sv : jetSVs){
    // basic kinematics
    fts.fill("sv_ptrel", sv->pt() / jet.pt());
    fts.fill("sv_erel", sv->energy() / jet.energy());
    fts.fill("sv_phirel", reco::deltaPhi(*sv, jet));
    fts.fill("sv_etarel", etasign * (sv->eta() - jet.eta()));
    fts.fill("sv_deltaR", reco::deltaR(*sv, jet));
    fts.fill("sv_pt", sv->pt());
    fts.fill("sv_abseta", std::abs(sv->eta()));
    fts.fill("sv_mass", sv->mass());

    fts.fill("sv_ptrel_log", catch_infs(std::log(sv->pt()/jet.pt()), -99));
    fts.fill("sv_erel_log", catch_infs(std::log(sv->energy()/jet.energy()), -99));
    fts.fill("sv_pt_log", catch_infs(std::log(sv->pt()), -99));

    // sv properties
    fts.fill("sv_ntracks", sv->numberOfDaughters());
    fts.fill("sv_chi2", sv->vertexChi2());
    fts.fill("sv_ndf", sv->vertexNdof());
    fts.fill("sv_normchi2", catch_infs(sv->vertexNormalizedChi2()));

    const auto &dxy = vertexDxy(*sv, *pv_);
    fts.fill("sv_dxy", dxy.value());
    fts.fill("sv_dxyerr", dxy.error());
    fts.fill("sv_dxysig", dxy.significance());

    const auto &d3d = vertexD3d(*sv, *pv_);
    fts.fill("sv_d3d", d3d.value());
    fts.fill("sv_d3derr", d3d.error());
    fts.fill("sv_d3dsig", d3d.significance());
    fts.fill("sv_costhetasvpv", vertexDdotP(*sv, *pv_));
  }

}

// define this as a plug-in
DEFINE_FWK_MODULE(DeepBoostedJetTagInfoProducer);

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
using namespace btagbtvdeep;

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

template <typename T>
class JetConstituentTableProducer : public edm::stream::EDProducer<> {
public:
  explicit JetConstituentTableProducer(const edm::ParameterSet &);
  ~JetConstituentTableProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  typedef reco::VertexCollection VertexCollection;
  //=====
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;

  //const std::string name_;
  const std::string name_;
  const std::string nameSV_;
  const std::string idx_name_;
  const std::string idx_nameSV_;
  const bool readBtag_;
  const double jet_radius_;

  edm::EDGetTokenT<edm::View<T>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<reco::CandidateView> cand_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;

  edm::Handle<VertexCollection> vtxs_;
  edm::Handle<reco::CandidateView> cands_;
  edm::Handle<SVCollection> svs_;
  edm::ESHandle<TransientTrackBuilder> track_builder_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> track_builder_token_;

  const reco::Vertex *pv_ = nullptr;
};

//
// constructors and destructor
//
template <typename T>
JetConstituentTableProducer<T>::JetConstituentTableProducer(const edm::ParameterSet &iConfig)
    : name_(iConfig.getParameter<std::string>("name")),
      nameSV_(iConfig.getParameter<std::string>("nameSV")),
      idx_name_(iConfig.getParameter<std::string>("idx_name")),
      idx_nameSV_(iConfig.getParameter<std::string>("idx_nameSV")),
      readBtag_(iConfig.getParameter<bool>("readBtag")),
      jet_radius_(iConfig.getParameter<double>("jet_radius")),
      jet_token_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      cand_token_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("candidates"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      track_builder_token_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))) {
  produces<nanoaod::FlatTable>(name_);
  produces<nanoaod::FlatTable>(nameSV_);
  produces<std::vector<reco::CandidatePtr>>();
}

template <typename T>
JetConstituentTableProducer<T>::~JetConstituentTableProducer() {}

template <typename T>
void JetConstituentTableProducer<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // elements in all these collections must have the same order!
  auto outCands = std::make_unique<std::vector<reco::CandidatePtr>>();
  auto outSVs = std::make_unique<std::vector<const reco::VertexCompositePtrCandidate *>>();
  std::vector<int> jetIdx_pf, jetIdx_sv, pfcandIdx, svIdx;
  // PF Cands
  std::vector<float> btagEtaRel, btagPtRatio, btagPParRatio, btagSip3dVal, btagSip3dSig, btagJetDistVal,
      btagDecayLenVal, cand_pt, cand_dzFromPV, cand_dxyFromPV, cand_dzErrFromPV, cand_dxyErrFromPV;
  // Secondary vertices
  std::vector<float> sv_mass, sv_pt, sv_ntracks, sv_chi2, sv_normchi2, sv_dxy, sv_dxysig, sv_d3d, sv_d3dsig,
      sv_costhetasvpv;
  std::vector<float> sv_ptrel, sv_phirel, sv_deltaR, sv_enratio;

  auto jets = iEvent.getHandle(jet_token_);
  iEvent.getByToken(vtx_token_, vtxs_);
  iEvent.getByToken(cand_token_, cands_);
  iEvent.getByToken(sv_token_, svs_);

  if (readBtag_) {
    track_builder_ = iSetup.getHandle(track_builder_token_);
  }

  for (unsigned i_jet = 0; i_jet < jets->size(); ++i_jet) {
    const auto &jet = jets->at(i_jet);
    math::XYZVector jet_dir = jet.momentum().Unit();
    GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());
    VertexDistance3D vdist;

    pv_ = &vtxs_->at(0);

    //////////////////////
    // Secondary Vertices
    std::vector<const reco::VertexCompositePtrCandidate *> jetSVs;
    std::vector<const reco::VertexCompositePtrCandidate *> allSVs;
    for (const auto &sv : *svs_) {
      // Factor in cuts in NanoAOD for indexing
      Measurement1D dl = vdist.distance(
          vtxs_->front(), VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
      if (dl.significance() > 3) {
        allSVs.push_back(&sv);
      }
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

    for (const auto &sv : jetSVs) {
      // auto svPtrs = svs_->ptrs();
      auto svInNewList = std::find(allSVs.begin(), allSVs.end(), sv);
      if (svInNewList == allSVs.end()) {
        // continue;
        svIdx.push_back(-1);
      } else {
        svIdx.push_back(svInNewList - allSVs.begin());
      }
      outSVs->push_back(sv);
      jetIdx_sv.push_back(i_jet);
      if (readBtag_ && !vtxs_->empty()) {
        // Jet independent
        sv_mass.push_back(sv->mass());
        sv_pt.push_back(sv->pt());

        sv_ntracks.push_back(sv->numberOfDaughters());
        sv_chi2.push_back(sv->vertexChi2());
        sv_normchi2.push_back(catch_infs_and_bound(sv->vertexChi2() / sv->vertexNdof(), 1000, -1000, 1000));
        const auto &dxy_meas = vertexDxy(*sv, *pv_);
        sv_dxy.push_back(dxy_meas.value());
        sv_dxysig.push_back(catch_infs_and_bound(dxy_meas.value() / dxy_meas.error(), 0, -1, 800));
        const auto &d3d_meas = vertexD3d(*sv, *pv_);
        sv_d3d.push_back(d3d_meas.value());
        sv_d3dsig.push_back(catch_infs_and_bound(d3d_meas.value() / d3d_meas.error(), 0, -1, 800));
        sv_costhetasvpv.push_back(vertexDdotP(*sv, *pv_));
        // Jet related
        sv_ptrel.push_back(sv->pt() / jet.pt());
        sv_phirel.push_back(reco::deltaPhi(*sv, jet));
        sv_deltaR.push_back(catch_infs_and_bound(std::fabs(reco::deltaR(*sv, jet_dir)) - 0.5, 0, -2, 0));
        sv_enratio.push_back(sv->energy() / jet.energy());
      }
    }

    // PF Cands
    std::vector<reco::CandidatePtr> const &daughters = jet.daughterPtrVector();

    for (const auto &cand : daughters) {
      auto candPtrs = cands_->ptrs();
      auto candInNewList = std::find(candPtrs.begin(), candPtrs.end(), cand);
      if (candInNewList == candPtrs.end()) {
        //std::cout << "Cannot find candidate : " << cand.id() << ", " << cand.key() << ", pt = " << cand->pt() << std::endl;
        continue;
      }
      outCands->push_back(cand);
      jetIdx_pf.push_back(i_jet);
      pfcandIdx.push_back(candInNewList - candPtrs.begin());
      cand_pt.push_back(cand->pt());
      auto const *packedCand = dynamic_cast<pat::PackedCandidate const *>(cand.get());
      if (packedCand && packedCand->hasTrackDetails()) {
        const reco::Track *track_ptr = &(packedCand->pseudoTrack());
        cand_dzFromPV.push_back(track_ptr->dz(pv_->position()));
        cand_dxyFromPV.push_back(track_ptr->dxy(pv_->position()));
        cand_dzErrFromPV.push_back(std::hypot(track_ptr->dzError(), pv_->zError()));
        cand_dxyErrFromPV.push_back(track_ptr->dxyError(pv_->position(), pv_->covariance()));
      } else {
        cand_dzFromPV.push_back(-1);
        cand_dxyFromPV.push_back(-1);
        cand_dzErrFromPV.push_back(-1);
        cand_dxyErrFromPV.push_back(-1);
      }

      if (readBtag_ && !vtxs_->empty()) {
        if (cand.isNull())
          continue;
        auto const *packedCand = dynamic_cast<pat::PackedCandidate const *>(cand.get());
        if (packedCand == nullptr)
          continue;
        if (packedCand && packedCand->hasTrackDetails()) {
          btagbtvdeep::TrackInfoBuilder trkinfo(track_builder_);
          trkinfo.buildTrackInfo(&(*packedCand), jet_dir, jet_ref_track_dir, vtxs_->at(0));
          btagEtaRel.push_back(trkinfo.getTrackEtaRel());
          btagPtRatio.push_back(trkinfo.getTrackPtRatio());
          btagPParRatio.push_back(trkinfo.getTrackPParRatio());
          btagSip3dVal.push_back(trkinfo.getTrackSip3dVal());
          btagSip3dSig.push_back(trkinfo.getTrackSip3dSig());
          btagJetDistVal.push_back(trkinfo.getTrackJetDistVal());
          // decay length
          const reco::Track *track_ptr = packedCand->bestTrack();
          reco::TransientTrack transient_track = track_builder_->build(track_ptr);
          double decayLength = -1;
          TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(
              transient_track.impactPointState(), *pv_, jet_ref_track_dir, transient_track.field());
          if (closest.isValid())
            decayLength = (closest.globalPosition() - RecoVertex::convertPos(pv_->position())).mag();
          else
            decayLength = -1;
          btagDecayLenVal.push_back(decayLength);
        } else {
          btagEtaRel.push_back(0);
          btagPtRatio.push_back(0);
          btagPParRatio.push_back(0);
          btagSip3dVal.push_back(0);
          btagSip3dSig.push_back(0);
          btagJetDistVal.push_back(0);
          btagDecayLenVal.push_back(0);
        }
      }
    }  // end jet loop
  }

  auto candTable = std::make_unique<nanoaod::FlatTable>(outCands->size(), name_, false);
  // We fill from here only stuff that cannot be created with the SimpleFlatTableProducer
  candTable->addColumn<int>(idx_name_, pfcandIdx, "Index in the candidate list");
  candTable->addColumn<int>("jetIdx", jetIdx_pf, "Index of the parent jet");
  if (readBtag_) {
    candTable->addColumn<float>("pt", cand_pt, "pt", 10);  // to check matchind down the line
    candTable->addColumn<float>("dzFromPV", cand_dzFromPV, "dzFromPV", 10);
    candTable->addColumn<float>("dxyFromPV", cand_dxyFromPV, "dxyFromPV", 10);
    candTable->addColumn<float>("dzErrFromPV", cand_dzErrFromPV, "dzErrFromPV", 10);
    candTable->addColumn<float>("dxyErrFromPV", cand_dxyErrFromPV, "dxyErrFromPV", 10);
    candTable->addColumn<float>("btagEtaRel", btagEtaRel, "btagEtaRel", 10);
    candTable->addColumn<float>("btagPtRatio", btagPtRatio, "btagPtRatio", 10);
    candTable->addColumn<float>("btagPParRatio", btagPParRatio, "btagPParRatio", 10);
    candTable->addColumn<float>("btagSip3dVal", btagSip3dVal, "btagSip3dVal", 10);
    candTable->addColumn<float>("btagSip3dSig", btagSip3dSig, "btagSip3dSig", 10);
    candTable->addColumn<float>("btagJetDistVal", btagJetDistVal, "btagJetDistVal", 10);
    candTable->addColumn<float>("btagDecayLenVal", btagDecayLenVal, "btagDecayLenVal", 10);
  }
  iEvent.put(std::move(candTable), name_);

  // SV table
  auto svTable = std::make_unique<nanoaod::FlatTable>(outSVs->size(), nameSV_, false);
  // We fill from here only stuff that cannot be created with the SimpleFlatTnameableProducer
  svTable->addColumn<int>("jetIdx", jetIdx_sv, "Index of the parent jet");
  svTable->addColumn<int>(idx_nameSV_, svIdx, "Index in the SV list");
  if (readBtag_) {
    svTable->addColumn<float>("mass", sv_mass, "SV mass", 10);
    svTable->addColumn<float>("pt", sv_pt, "SV pt", 10);
    svTable->addColumn<float>("ntracks", sv_ntracks, "Number of tracks associated to SV", 10);
    svTable->addColumn<float>("chi2", sv_chi2, "chi2", 10);
    svTable->addColumn<float>("normchi2", sv_normchi2, "chi2/ndof", 10);
    svTable->addColumn<float>("dxy", sv_dxy, "", 10);
    svTable->addColumn<float>("dxysig", sv_dxysig, "", 10);
    svTable->addColumn<float>("d3d", sv_d3d, "", 10);
    svTable->addColumn<float>("d3dsig", sv_d3dsig, "", 10);
    svTable->addColumn<float>("costhetasvpv", sv_costhetasvpv, "", 10);
    // Jet related
    svTable->addColumn<float>("phirel", sv_phirel, "DeltaPhi(sv, jet)", 10);
    svTable->addColumn<float>("ptrel", sv_ptrel, "pT relative to parent jet", 10);
    svTable->addColumn<float>("deltaR", sv_deltaR, "dR from parent jet", 10);
    svTable->addColumn<float>("enration", sv_enratio, "energy relative to parent jet", 10);
  }
  iEvent.put(std::move(svTable), nameSV_);

  iEvent.put(std::move(outCands));
}

template <typename T>
void JetConstituentTableProducer<T>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("name", "JetPFCands");
  desc.add<std::string>("nameSV", "JetSV");
  desc.add<std::string>("idx_name", "candIdx");
  desc.add<std::string>("idx_nameSV", "svIdx");
  desc.add<double>("jet_radius", true);
  desc.add<bool>("readBtag", true);
  desc.add<edm::InputTag>("jets", edm::InputTag("slimmedJetsAK8"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("slimmedSecondaryVertices"));
  descriptions.addWithDefaultLabel(desc);
}

typedef JetConstituentTableProducer<pat::Jet> PatJetConstituentTableProducer;
typedef JetConstituentTableProducer<reco::GenJet> GenJetConstituentTableProducer;

DEFINE_FWK_MODULE(PatJetConstituentTableProducer);
DEFINE_FWK_MODULE(GenJetConstituentTableProducer);

// -*- C++ -*-
//
// Package:    HLTrigger/NGTScouting
// Class:      HLTV0TableProducer
//
/**\class HLTV0TableProducer HLTV0TableProducer.cc HLTrigger/NGTScouting/plugins/HLTV0TableProducer.cc

 Description: produces a nanoaod::FlatTable with the HLT V0 candidates (K0s or Lambda)
              reconstructed by the V0Producer (reco::VertexCompositeCandidateCollection).
              Displacement and pointing variables are computed w.r.t. the beamspot
              (transverse) and w.r.t. the leading primary vertex (3D).
*/

#include <cmath>
#include <limits>
#include <memory>

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

class HLTV0TableProducer : public edm::global::EDProducer<> {
public:
  explicit HLTV0TableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> v0Token_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  const std::string name_;
  const std::string doc_;
};

HLTV0TableProducer::HLTV0TableProducer(const edm::ParameterSet& params)
    : skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
      v0Token_(consumes(params.getParameter<edm::InputTag>("src"))),
      bsToken_(consumes(params.getParameter<edm::InputTag>("beamSpot"))),
      pvToken_(consumes(params.getParameter<edm::InputTag>("primaryVertices"))),
      name_(params.getParameter<std::string>("name")),
      doc_(params.getParameter<std::string>("doc")) {
  produces<nanoaod::FlatTable>();
}

void HLTV0TableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  static constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

  const auto v0Handle = iEvent.getHandle(v0Token_);
  if (!v0Handle.isValid()) {
    if (!skipNonExistingSrc_) {
      // rethrows the ProductNotFound exception
      v0Handle.product();
    }
    edm::LogWarning("HLTV0TableProducer")
        << "Invalid handle for V0 input collection, producing an empty " << name_ << " table";
  }
  const size_t nV0s = v0Handle.isValid() ? v0Handle->size() : 0;

  const auto bsHandle = iEvent.getHandle(bsToken_);
  const reco::BeamSpot* bs = bsHandle.isValid() ? bsHandle.product() : nullptr;

  const auto pvHandle = iEvent.getHandle(pvToken_);
  const reco::Vertex* pv0 = (pvHandle.isValid() && !pvHandle->empty()) ? &pvHandle->front() : nullptr;

  std::vector<float> v_pt(nV0s, kNaN), v_eta(nV0s, kNaN), v_phi(nV0s, kNaN), v_mass(nV0s, kNaN);
  std::vector<int> v_pdgId(nV0s, 0);
  std::vector<float> v_x(nV0s, kNaN), v_y(nV0s, kNaN), v_z(nV0s, kNaN);
  std::vector<float> v_chi2(nV0s, kNaN), v_ndof(nV0s, kNaN);
  std::vector<float> v_lxy(nV0s, kNaN), v_lxySig(nV0s, kNaN), v_cosThetaXY(nV0s, kNaN);
  std::vector<float> v_l3d(nV0s, kNaN), v_l3dSig(nV0s, kNaN), v_cosTheta3D(nV0s, kNaN);
  std::vector<float> v_posPt(nV0s, kNaN), v_posEta(nV0s, kNaN), v_posPhi(nV0s, kNaN);
  std::vector<float> v_negPt(nV0s, kNaN), v_negEta(nV0s, kNaN), v_negPhi(nV0s, kNaN);
  std::vector<int16_t> v_posTrkIdx(nV0s, -1), v_negTrkIdx(nV0s, -1);
  std::vector<uint8_t> v_posNPixelHits(nV0s, 0), v_negNPixelHits(nV0s, 0);
  std::vector<uint8_t> v_posNStripHits(nV0s, 0), v_negNStripHits(nV0s, 0);
  std::vector<float> v_posIPSigXY(nV0s, kNaN), v_negIPSigXY(nV0s, kNaN);

  VertexDistanceXY vdistXY;
  VertexDistance3D vdist3D;

  for (size_t i = 0; i < nV0s; ++i) {
    const auto& v0 = (*v0Handle)[i];

    v_pt[i] = v0.pt();
    v_eta[i] = v0.eta();
    v_phi[i] = v0.phi();
    v_mass[i] = v0.mass();
    v_pdgId[i] = v0.pdgId();
    v_x[i] = v0.vx();
    v_y[i] = v0.vy();
    v_z[i] = v0.vz();
    v_chi2[i] = v0.vertexNormalizedChi2();
    v_ndof[i] = v0.vertexNdof();

    reco::Vertex::Error v0Cov;
    v0.fillVertexCovariance(v0Cov);
    const VertexState v0State(RecoVertex::convertPos(v0.vertex()), RecoVertex::convertError(v0Cov));

    // transverse displacement w.r.t. the beamspot, evaluated at the z of the V0 (accounts for the beamline tilt)
    if (bs != nullptr) {
      const reco::Vertex bsVertex(
          reco::Vertex::Point(bs->x(v0.vz()), bs->y(v0.vz()), v0.vz()), bs->rotatedCovariance3D(), 0., 0., 0);
      const Measurement1D dxy = vdistXY.distance(bsVertex, v0State);
      v_lxy[i] = dxy.value();
      v_lxySig[i] = dxy.significance();

      const double dx = v0.vx() - bsVertex.x();
      const double dy = v0.vy() - bsVertex.y();
      const double rxy = std::hypot(dx, dy);
      if (rxy > 0. && v0.pt() > 0.)
        v_cosThetaXY[i] = (dx * v0.px() + dy * v0.py()) / (rxy * v0.pt());
    }

    // 3D displacement w.r.t. the leading primary vertex
    if (pv0 != nullptr) {
      const Measurement1D d3d = vdist3D.distance(*pv0, v0State);
      v_l3d[i] = d3d.value();
      v_l3dSig[i] = d3d.significance();

      const double dx = v0.vx() - pv0->x();
      const double dy = v0.vy() - pv0->y();
      const double dz = v0.vz() - pv0->z();
      const double r3d = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (r3d > 0. && v0.p() > 0.)
        v_cosTheta3D[i] = (dx * v0.px() + dy * v0.py() + dz * v0.pz()) / (r3d * v0.p());
    }

    // daughters (two oppositely charged tracks)
    for (size_t id = 0; id < v0.numberOfDaughters(); ++id) {
      const reco::Candidate* dau = v0.daughter(id);
      if (dau == nullptr || dau->charge() == 0)
        continue;
      int16_t trkIdx = -1;
      uint8_t nPixelHits = 0, nStripHits = 0;
      float ipSigXY = kNaN;
      if (const auto* rcc = dynamic_cast<const reco::RecoChargedCandidate*>(dau); rcc && rcc->track().isNonnull()) {
        const auto& trk = rcc->track();
        trkIdx = static_cast<int16_t>(trk.key());
        // in phase-2 the Outer Tracker hits are counted as strip hits
        nPixelHits = static_cast<uint8_t>(trk->hitPattern().numberOfValidPixelHits());
        nStripHits = static_cast<uint8_t>(trk->hitPattern().numberOfValidStripHits());
        if (bs != nullptr && trk->dxyError() > 0.)
          ipSigXY = std::abs(trk->dxy(bs->position(trk->vz()))) / trk->dxyError();
      }
      if (dau->charge() > 0) {
        v_posPt[i] = dau->pt();
        v_posEta[i] = dau->eta();
        v_posPhi[i] = dau->phi();
        v_posTrkIdx[i] = trkIdx;
        v_posNPixelHits[i] = nPixelHits;
        v_posNStripHits[i] = nStripHits;
        v_posIPSigXY[i] = ipSigXY;
      } else {
        v_negPt[i] = dau->pt();
        v_negEta[i] = dau->eta();
        v_negPhi[i] = dau->phi();
        v_negTrkIdx[i] = trkIdx;
        v_negNPixelHits[i] = nPixelHits;
        v_negNStripHits[i] = nStripHits;
        v_negIPSigXY[i] = ipSigXY;
      }
    }
  }

  auto table = std::make_unique<nanoaod::FlatTable>(nV0s, name_, /*singleton*/ false);
  table->setDoc(doc_);
  table->addColumn<float>("pt", v_pt, "p_T (GeV)", 10);
  table->addColumn<float>("eta", v_eta, "#eta", 12);
  table->addColumn<float>("phi", v_phi, "#phi (rad)", 12);
  table->addColumn<float>("mass", v_mass, "invariant mass (GeV)", 14);
  table->addColumn<int>("pdgId", v_pdgId, "PDG id (310: K0s, +/-3122: Lambda/anti-Lambda)");
  table->addColumn<float>("x", v_x, "V0 vertex x position (cm)", 10);
  table->addColumn<float>("y", v_y, "V0 vertex y position (cm)", 10);
  table->addColumn<float>("z", v_z, "V0 vertex z position (cm)", 14);
  table->addColumn<float>("chi2", v_chi2, "V0 vertex reduced chi2, i.e. chi2/ndof", 8);
  table->addColumn<float>("ndof", v_ndof, "V0 vertex number of degrees of freedom", 8);
  table->addColumn<float>("lxy", v_lxy, "transverse decay length w.r.t. the beamspot (cm)", 10);
  table->addColumn<float>("lxySig", v_lxySig, "transverse decay length significance w.r.t. the beamspot", 10);
  table->addColumn<float>(
      "cosThetaXY", v_cosThetaXY, "cosine of the transverse pointing angle w.r.t. the beamspot", 12);
  table->addColumn<float>("l3d", v_l3d, "3D decay length w.r.t. the leading primary vertex (cm)", 10);
  table->addColumn<float>("l3dSig", v_l3dSig, "3D decay length significance w.r.t. the leading primary vertex", 10);
  table->addColumn<float>(
      "cosTheta3D", v_cosTheta3D, "cosine of the 3D pointing angle w.r.t. the leading primary vertex", 12);
  table->addColumn<float>("posPt", v_posPt, "positive daughter p_T (GeV)", 10);
  table->addColumn<float>("posEta", v_posEta, "positive daughter #eta", 12);
  table->addColumn<float>("posPhi", v_posPhi, "positive daughter #phi (rad)", 12);
  table->addColumn<int16_t>(
      "posTrkIdx", v_posTrkIdx, "index of the positive daughter in the input track collection (-1 if unavailable)");
  table->addColumn<uint8_t>(
      "posNPixelHits", v_posNPixelHits, "number of valid pixel hits of the positive daughter track");
  table->addColumn<uint8_t>("posNStripHits",
                            v_posNStripHits,
                            "number of valid strip (phase-2: Outer Tracker) hits of the positive daughter track");
  table->addColumn<float>("posIPSigXY",
                          v_posIPSigXY,
                          "transverse impact parameter significance of the positive daughter track w.r.t. the beamspot",
                          10);
  table->addColumn<float>("negPt", v_negPt, "negative daughter p_T (GeV)", 10);
  table->addColumn<float>("negEta", v_negEta, "negative daughter #eta", 12);
  table->addColumn<float>("negPhi", v_negPhi, "negative daughter #phi (rad)", 12);
  table->addColumn<int16_t>(
      "negTrkIdx", v_negTrkIdx, "index of the negative daughter in the input track collection (-1 if unavailable)");
  table->addColumn<uint8_t>(
      "negNPixelHits", v_negNPixelHits, "number of valid pixel hits of the negative daughter track");
  table->addColumn<uint8_t>("negNStripHits",
                            v_negNStripHits,
                            "number of valid strip (phase-2: Outer Tracker) hits of the negative daughter track");
  table->addColumn<float>("negIPSigXY",
                          v_negIPSigXY,
                          "transverse impact parameter significance of the negative daughter track w.r.t. the beamspot",
                          10);

  iEvent.put(std::move(table));
}

void HLTV0TableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to produce an empty table on absent input product");
  desc.add<edm::InputTag>("src", edm::InputTag("hltGeneralV0Candidates", "Kshort"))
      ->setComment("reco::VertexCompositeCandidateCollection input collection");
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"))
      ->setComment("beamspot used for the transverse displacement");
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("hltOfflinePrimaryVertices"))
      ->setComment("primary vertices: the leading one is used for the 3D displacement");
  desc.add<std::string>("name", "hltKshort")->setComment("name of the flat table output");
  desc.add<std::string>("doc", "HLT K0s candidates")->setComment("a few words of documentation");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HLTV0TableProducer);

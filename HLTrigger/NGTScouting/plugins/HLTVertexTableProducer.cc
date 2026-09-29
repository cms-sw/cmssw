#include <memory>

// user include files
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"

//
// class declaration
//

class HLTVertexTableProducer : public edm::stream::EDProducer<> {
public:
  explicit HLTVertexTableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  const bool skipNonExistingSrc_;
  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvs_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfc_;
  const edm::EDGetTokenT<edm::ValueMap<float>> pvsScore_;
  const edm::EDGetTokenT<edm::View<reco::VertexCompositePtrCandidate>> svs_;
  const StringCutObjectSelector<reco::Vertex> goodPvCut_;
  const std::string goodPvCutString_;
  const StringCutObjectSelector<reco::VertexCompositePtrCandidate> goodSvCut_;
  const std::string goodSvCutString_;
  const bool usePF_;
  const bool doSVs_;
  const std::string pvName_;
  const std::string svName_;
  const std::string svDoc_;
  const double dlenMin_, dlenSigMin_;
};

//
// constructors
//

HLTVertexTableProducer::HLTVertexTableProducer(const edm::ParameterSet& params)
    : skipNonExistingSrc_(params.getParameter<bool>("skipNonExistingSrc")),
      pvs_(consumes<std::vector<reco::Vertex>>(params.getParameter<edm::InputTag>("pvSrc"))),
      pfc_(consumes<reco::PFCandidateCollection>(params.getParameter<edm::InputTag>("pfSrc"))),
      pvsScore_(consumes<edm::ValueMap<float>>(params.getParameter<edm::InputTag>("pvSrc"))),
      svs_(consumes<edm::View<reco::VertexCompositePtrCandidate>>(params.getParameter<edm::InputTag>("svSrc"))),
      goodPvCut_(params.getParameter<std::string>("goodPvCut"), true),
      goodPvCutString_(params.getParameter<std::string>("goodPvCut")),
      goodSvCut_(params.getParameter<std::string>("goodSvCut"), true),
      goodSvCutString_(params.getParameter<std::string>("goodSvCut")),
      usePF_(params.getParameter<bool>("usePF")),
      doSVs_(params.getParameter<bool>("doSVs")),
      pvName_(params.getParameter<std::string>("pvName")),
      svName_(params.getParameter<std::string>("svName")),
      svDoc_(params.getParameter<std::string>("svDoc")),
      dlenMin_(params.getParameter<double>("dlenMin")),
      dlenSigMin_(params.getParameter<double>("dlenSigMin")) {
  produces<nanoaod::FlatTable>("PV");
  if (doSVs_)
    produces<nanoaod::FlatTable>("SV");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HLTVertexTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //vertex collections (PVs and SVs)
  auto pvsIn = iEvent.getHandle(pvs_);
  auto pvScoreIn = iEvent.getHandle(pvsScore_);
  const size_t nPVs = pvsIn.isValid() ? (*pvsIn).size() : 0;
  auto svsIn = iEvent.getHandle(svs_);
  const size_t nSVs = svsIn.isValid() ? (*svsIn).size() : 0;

  static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

  std::vector<float> v_pv_ndof(nPVs, default_value);
  std::vector<float> v_pv_chi2(nPVs, default_value);
  std::vector<float> v_pv_x(nPVs, default_value);
  std::vector<float> v_pv_y(nPVs, default_value);
  std::vector<float> v_pv_z(nPVs, default_value);
  std::vector<float> v_pv_xError(nPVs, default_value);
  std::vector<float> v_pv_yError(nPVs, default_value);
  std::vector<float> v_pv_zError(nPVs, default_value);
  std::vector<uint8_t> v_pv_is_good(nPVs, 0u);
  std::vector<uint8_t> v_pv_nTracks(nPVs, 0u);
  std::vector<float> v_pv_score(nPVs, default_value);
  std::vector<float> v_pv_sumpt2(nPVs, default_value);
  std::vector<float> v_pv_sumpx(nPVs, default_value);
  std::vector<float> v_pv_sumpy(nPVs, default_value);

  std::vector<int8_t> v_sv_charge(nSVs, 0);
  std::vector<float> v_sv_chi2(nSVs, default_value);
  std::vector<float> v_sv_dlen(nSVs, default_value);
  std::vector<float> v_sv_dlenSig(nSVs, default_value);
  std::vector<float> v_sv_dxy(nSVs, default_value);
  std::vector<float> v_sv_dxySig(nSVs, default_value);
  std::vector<float> v_sv_eta(nSVs, default_value);
  std::vector<float> v_sv_mass(nSVs, default_value);
  std::vector<float> v_sv_ndof(nSVs, default_value);
  std::vector<uint8_t> v_sv_is_good(nSVs, 0u);
  std::vector<uint8_t> v_sv_ntracks(nSVs, 0u);
  std::vector<float> v_sv_pAngle(nSVs, default_value);
  std::vector<float> v_sv_phi(nSVs, default_value);
  std::vector<float> v_sv_pt(nSVs, default_value);
  std::vector<float> v_sv_x(nSVs, default_value);
  std::vector<float> v_sv_y(nSVs, default_value);
  std::vector<float> v_sv_z(nSVs, default_value);

  if (pvsIn.isValid() || !(this->skipNonExistingSrc_)) {
    const auto& pvs = *pvsIn;

    auto pfcIn = iEvent.getHandle(pfc_);
    const bool isPfcValid = pfcIn.isValid();

    for (size_t i = 0; i < nPVs; ++i) {
      const auto& pv = pvs[i];
      const auto& pos = pv.position();

      v_pv_ndof[i] = pv.ndof();
      v_pv_chi2[i] = pv.normalizedChi2();
      v_pv_x[i] = pv.x();
      v_pv_y[i] = pv.y();
      v_pv_z[i] = pv.z();
      v_pv_xError[i] = pv.xError();
      v_pv_yError[i] = pv.yError();
      v_pv_zError[i] = pv.zError();
      v_pv_nTracks[i] = pv.nTracks();
      v_pv_is_good[i] = goodPvCut_(pv);

      if (pvScoreIn.isValid() || !(this->skipNonExistingSrc_)) {
        const auto& pvsScoreProd = *pvScoreIn;
        v_pv_score[i] = pvsScoreProd.get(pvsIn.id(), i);
      }

      float sumpt2 = 0.f, sumpx = 0.f, sumpy = 0.f;

      if (usePF_) {
        if (isPfcValid || !(this->skipNonExistingSrc_)) {
          for (const auto& obj : *pfcIn) {
            if (obj.charge() == 0 || !obj.trackRef().isNonnull())
              continue;

            const auto dz = std::abs(obj.trackRef()->dz(pos));
            if (dz >= 0.2)
              continue;

            bool isClosest = true;
            for (size_t j = 0; j < nPVs; ++j) {
              if (j == i)
                continue;
              const auto dz_j = std::abs(obj.trackRef()->dz(pvs[j].position()));
              if (dz_j < dz) {
                isClosest = false;
                break;
              }
            }

            if (isClosest) {
              const float pt = obj.pt();
              sumpt2 += pt * pt;
              sumpx += obj.px();
              sumpy += obj.py();
            }
          }
        } else {
          edm::LogWarning("HLTVertexTableProducer")
              << " Invalid handle for " << pvName_ << " in PF candidate input collection";
        }
      } else {
        // Loop over tracks used in PV fit
        for (auto t = pv.tracks_begin(); t != pv.tracks_end(); ++t) {
          const auto& trk = **t;  // trk is a reco::TrackBase
          const float pt = trk.pt();
          sumpt2 += pt * pt;
          sumpx += trk.px();
          sumpy += trk.py();
        }
      }
      v_pv_sumpt2[i] = sumpt2;
      v_pv_sumpx[i] = sumpx;
      v_pv_sumpy[i] = sumpy;
    }

    if (doSVs_ && (nPVs > 0)) {
      if (svsIn.isValid() || !(this->skipNonExistingSrc_)) {
        const auto& svs = *svsIn;

        VertexDistance3D vdist;
        VertexDistanceXY vdistXY;
        const auto& PV0 = pvsIn->front();
        double pv0_x = PV0.x(), pv0_y = PV0.y(), pv0_z = PV0.z();

        for (size_t i = 0; i < nSVs; ++i) {
          auto const& sv = svs[i];

          Measurement1D dl = vdist.distance(
              PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));

          v_sv_chi2[i] = sv.vertexNormalizedChi2();
          v_sv_ndof[i] = sv.vertexNdof();
          v_sv_ntracks[i] = sv.numberOfDaughters();
          v_sv_pt[i] = sv.pt();
          v_sv_eta[i] = sv.eta();
          v_sv_phi[i] = sv.phi();
          v_sv_mass[i] = sv.mass();
          v_sv_dlen[i] = dl.value();
          v_sv_dlenSig[i] = dl.significance();
          v_sv_is_good[i] = goodSvCut_(sv) && (dl.value() > dlenMin_) && (dl.significance() > dlenSigMin_);

          double x = sv.vx(), y = sv.vy(), z = sv.vz();
          v_sv_x[i] = x;
          v_sv_y[i] = y;
          v_sv_z[i] = z;

          double dx = (pv0_x - x), dy = (pv0_y - y), dz = (pv0_z - z);
          double pdotv = (dx * sv.px() + dy * sv.py() + dz * sv.pz()) / sv.p() / sqrt(dx * dx + dy * dy + dz * dz);
          v_sv_pAngle[i] = std::acos(pdotv);

          Measurement1D d2d = vdistXY.distance(
              PV0, VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
          v_sv_dxy[i] = d2d.value();
          v_sv_dxySig[i] = d2d.significance();

          int sum_charge = 0;
          for (size_t id = 0; id < sv.numberOfDaughters(); ++id) {
            const reco::Candidate* daughter = sv.daughter(id);
            sum_charge += daughter->charge();
          }
          v_sv_charge[i] = sum_charge;
        }
      } else {
        edm::LogWarning("HLTVertexTableProducer")
            << " Invalid handle for " << svName_ << " in secondary vertex input collection";
      }
    }
  } else {
    edm::LogWarning("HLTVertexTableProducer")
        << " Invalid handle for " << pvName_ << " in primary vertex input collection";
  }

  // table for all primary vertices
  auto pvTable = std::make_unique<nanoaod::FlatTable>(nPVs, pvName_, /*singleton*/ false);
  pvTable->addColumn<float>("ndof", v_pv_ndof, "primary vertex number of degrees of freedom", 8);
  pvTable->addColumn<float>("chi2", v_pv_chi2, "primary vertex reduced chi2", 8);
  pvTable->addColumn<float>("x", v_pv_x, "primary vertex x coordinate", 10);
  pvTable->addColumn<float>("y", v_pv_y, "primary vertex y coordinate", 10);
  pvTable->addColumn<float>("z", v_pv_z, "primary vertex z coordinate", 16);
  pvTable->addColumn<float>("xError", v_pv_xError, "primary vertex error in x coordinate", 10);
  pvTable->addColumn<float>("yError", v_pv_yError, "primary vertex error in y coordinate", 10);
  pvTable->addColumn<float>("zError", v_pv_zError, "primary vertex error in z coordinate", 16);
  pvTable->addColumn<uint8_t>(
      "isGood", v_pv_is_good, "whether the primary vertex passes selection: " + goodPvCutString_ + ")");
  pvTable->addColumn<uint8_t>("nTracks", v_pv_nTracks, "primary vertex number of associated tracks");
  pvTable->addColumn<float>("score", v_pv_score, "primary vertex score, i.e. sum pt2 of clustered objects", 8);
  pvTable->addColumn<float>(
      "sumpt2", v_pv_sumpt2, "sum pt2 of pf charged candidates within dz=0.2 for the main primary vertex", 10);
  pvTable->addColumn<float>(
      "sumpx", v_pv_sumpx, "sum px of pf charged candidates within dz=0.2 for the main primary vertex", 10);
  pvTable->addColumn<float>(
      "sumpy", v_pv_sumpy, "sum py of pf charged candidates within dz=0.2 for the main primary vertex", 10);

  iEvent.put(std::move(pvTable), "PV");

  if (doSVs_) {
    // table for all secondary vertices
    auto svTable = std::make_unique<nanoaod::FlatTable>(nSVs, svName_, /*singleton*/ false);
    svTable->setDoc(svDoc_);
    svTable->addColumn<float>("dlen", v_sv_dlen, "decay length in cm", 10);
    svTable->addColumn<float>("dlenSig", v_sv_dlenSig, "decay length significance", 10);
    svTable->addColumn<float>("dxy", v_sv_dxy, "2D decay length in cm", 10);
    svTable->addColumn<float>("dxySig", v_sv_dxySig, "2D decay length significance", 10);
    svTable->addColumn<float>("pAngle", v_sv_pAngle, "pointing angle, i.e. acos(p_SV * (SV - PV)) ", 10);
    svTable->addColumn<int16_t>("charge", v_sv_charge, "sum of the charge of the SV tracks", 10);
    svTable->addColumn<float>("x", v_sv_x, "secondary vertex X position, in cm", 10);
    svTable->addColumn<float>("y", v_sv_y, "secondary vertex Y position, in cm", 10);
    svTable->addColumn<float>("z", v_sv_z, "secondary vertex Z position, in cm", 14);
    svTable->addColumn<float>("ndof", v_sv_ndof, "number of degrees of freedom", 8);
    svTable->addColumn<float>("chi2", v_sv_chi2, "reduced chi2, i.e. chi/ndof", 8);
    svTable->addColumn<uint8_t>("ntracks", v_sv_ntracks, "number of tracks");
    svTable->addColumn<uint8_t>(
        "isGood", v_sv_is_good, "whether the secondary vertex passes selection: " + goodSvCutString_ + ")");
    svTable->addColumn<float>("pt", v_sv_pt, "pt", 10);
    svTable->addColumn<float>("eta", v_sv_eta, "eta", 12);
    svTable->addColumn<float>("phi", v_sv_phi, "phi", 12);
    svTable->addColumn<float>("mass", v_sv_mass, "mass", 10);

    iEvent.put(std::move(svTable), "SV");
  }
}

// ------------ fill 'descriptions' with the allowed parameters for the module ------------
void HLTVertexTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // general flags
  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to skip producing the table on absent input product");
  desc.add<bool>("usePF", true)
      ->setComment("if true, use PF candidate-based association; if false, use only tracks used in PV fit");
  desc.add<bool>("doSVs", true)->setComment("if true, produce secondary vertex table");

  // Primary Vertices
  desc.add<std::string>("pvName", "hltPrimaryVertex")->setComment("name of the flat table ouput");
  desc.add<edm::InputTag>("pvSrc", edm::InputTag("hltOfflinePrimaryVertices"))
      ->setComment("std::vector<reco::Vertex> and ValueMap<float> primary vertex input collections");
  desc.add<edm::InputTag>("pfSrc", edm::InputTag("hltParticleFlowTmp"))
      ->setComment("reco::PFCandidateCollection PF candidates input collections");
  desc.add<std::string>("goodPvCut", "")->setComment("selection on the primary vertex");

  // Secondary Vertices
  desc.add<std::string>("svName", "hltSecondaryVertex")->setComment("name of the flat table ouput");
  desc.add<std::string>("svDoc", "secondary vertices from IVF algorithm")->setComment("a few words of documentation");
  desc.add<edm::InputTag>("svSrc", edm::InputTag("hltDeepInclusiveSecondaryVerticesPF"))
      ->setComment("reco::VertexCompositePtrCandidate compatible secondary vertex input collection");
  desc.add<std::string>("goodSvCut", "")
      ->setComment("selection on the secondary vertex on top of dlen and dlenSig cuts");
  desc.add<double>("dlenMin", 0.0)->setComment("minimum value of dl to call a secondary vertex good");
  desc.add<double>("dlenSigMin", 3.0)->setComment("minimum value of dl significance to call a secondary vertex good");

  descriptions.addWithDefaultLabel(desc);
}

// ------------ define this as a plug-in ------------
DEFINE_FWK_MODULE(HLTVertexTableProducer);

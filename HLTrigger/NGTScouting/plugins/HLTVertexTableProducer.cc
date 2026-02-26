#include <memory>

// user include files
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
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
  const StringCutObjectSelector<reco::Vertex> goodPvCut_;
  const std::string goodPvCutString_;
  const bool usePF_;
  const std::string pvName_;
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
      goodPvCut_(params.getParameter<std::string>("goodPvCut"), true),
      goodPvCutString_(params.getParameter<std::string>("goodPvCut")),
      usePF_(params.getParameter<bool>("usePF")),
      pvName_(params.getParameter<std::string>("pvName")),
      dlenMin_(params.getParameter<double>("dlenMin")),
      dlenSigMin_(params.getParameter<double>("dlenSigMin")) {
  produces<nanoaod::FlatTable>("PV");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HLTVertexTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //vertex collection
  auto pvsIn = iEvent.getHandle(pvs_);
  auto pvScoreIn = iEvent.getHandle(pvsScore_);
  const size_t nPVs = pvsIn.isValid() ? (*pvsIn).size() : 0;

  static constexpr float default_value = std::numeric_limits<float>::quiet_NaN();

  std::vector<float> v_ndof(nPVs, default_value);
  std::vector<float> v_chi2(nPVs, default_value);
  std::vector<float> v_x(nPVs, default_value);
  std::vector<float> v_y(nPVs, default_value);
  std::vector<float> v_z(nPVs, default_value);
  std::vector<float> v_xError(nPVs, default_value);
  std::vector<float> v_yError(nPVs, default_value);
  std::vector<float> v_zError(nPVs, default_value);
  std::vector<uint8_t> v_is_good(nPVs, 0);
  std::vector<uint8_t> v_nTracks(nPVs, 0);
  std::vector<float> v_pv_score(nPVs, default_value);
  std::vector<float> v_pv_sumpt2(nPVs, default_value);
  std::vector<float> v_pv_sumpx(nPVs, default_value);
  std::vector<float> v_pv_sumpy(nPVs, default_value);

  if (pvsIn.isValid() || !(this->skipNonExistingSrc_)) {
    const auto& pvs = *pvsIn;

    auto pfcIn = iEvent.getHandle(pfc_);
    const bool isPfcValid = pfcIn.isValid();

    for (size_t i = 0; i < nPVs; ++i) {
      const auto& pv = pvs[i];
      const auto& pos = pv.position();

      v_ndof[i] = pv.ndof();
      v_chi2[i] = pv.normalizedChi2();
      v_x[i] = pv.x();
      v_y[i] = pv.y();
      v_z[i] = pv.z();
      v_xError[i] = pv.xError();
      v_yError[i] = pv.yError();
      v_zError[i] = pv.zError();
      v_nTracks[i] = pv.nTracks();
      v_is_good[i] = goodPvCut_(pv);

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
  } else {
    edm::LogWarning("HLTVertexTableProducer")
        << " Invalid handle for " << pvName_ << " in primary vertex input collection";
  }

  //table for all primary vertices
  auto pvTable = std::make_unique<nanoaod::FlatTable>(nPVs, pvName_, true);
  pvTable->addColumn<float>("ndof", v_ndof, "primary vertex number of degrees of freedom", 8);
  pvTable->addColumn<float>("chi2", v_chi2, "primary vertex reduced chi2", 8);
  pvTable->addColumn<float>("x", v_x, "primary vertex x coordinate", 10);
  pvTable->addColumn<float>("y", v_y, "primary vertex y coordinate", 10);
  pvTable->addColumn<float>("z", v_z, "primary vertex z coordinate", 16);
  pvTable->addColumn<float>("xError", v_xError, "primary vertex error in x coordinate", 10);
  pvTable->addColumn<float>("yError", v_yError, "primary vertex error in y coordinate", 10);
  pvTable->addColumn<float>("zError", v_zError, "primary vertex error in z coordinate", 16);
  pvTable->addColumn<uint8_t>(
      "isGood", v_is_good, "wheter the primary vertex passes selection: " + goodPvCutString_ + ")");
  pvTable->addColumn<uint8_t>("nTracks", v_nTracks, "primary vertex number of associated tracks");
  pvTable->addColumn<float>("score", v_pv_score, "primary vertex score, i.e. sum pt2 of clustered objects", 8);
  pvTable->addColumn<float>(
      "sumpt2", v_pv_sumpt2, "sum pt2 of pf charged candidates within dz=0.2 for the main primary vertex", 10);
  pvTable->addColumn<float>(
      "sumpx", v_pv_sumpx, "sum px of pf charged candidates within dz=0.2 for the main primary vertex", 10);
  pvTable->addColumn<float>(
      "sumpy", v_pv_sumpy, "sum py of pf charged candidates within dz=0.2 for the main primary vertex", 10);

  iEvent.put(std::move(pvTable), "PV");
}

// ------------ fill 'descriptions' with the allowed parameters for the module ------------
void HLTVertexTableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to skip producing the table on absent input product");
  desc.add<std::string>("pvName")->setComment("name of the flat table ouput");
  desc.add<edm::InputTag>("pvSrc")->setComment(
      "std::vector<reco::Vertex> and ValueMap<float> primary vertex input collections");
  desc.add<bool>("usePF", true)
      ->setComment("if true, use PF candidate-based association; if false, use only tracks used in PV fit");
  desc.add<edm::InputTag>("pfSrc")->setComment("reco::PFCandidateCollection PF candidates input collections");
  desc.add<std::string>("goodPvCut")->setComment("selection on the primary vertex");
  desc.add<double>("dlenMin")->setComment("minimum value of dl to select secondary vertex");
  desc.add<double>("dlenSigMin")->setComment("minimum value of dl significance to select secondary vertex");
  descriptions.addWithDefaultLabel(desc);
}

// ------------ define this as a plug-in ------------
DEFINE_FWK_MODULE(HLTVertexTableProducer);

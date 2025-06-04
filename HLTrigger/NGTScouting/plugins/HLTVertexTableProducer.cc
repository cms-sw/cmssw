#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

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
  const edm::EDGetTokenT<std::vector<reco::Vertex>> pvs_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfc_;
  const edm::EDGetTokenT<edm::ValueMap<float>> pvsScore_;
  const StringCutObjectSelector<reco::Vertex> goodPvCut_;
  const std::string goodPvCutString_;
  const std::string pvName_;
  const double dlenMin_, dlenSigMin_;
};

//
// constructors
//

HLTVertexTableProducer::HLTVertexTableProducer(const edm::ParameterSet& params)
    : pvs_(consumes<std::vector<reco::Vertex>>(params.getParameter<edm::InputTag>("pvSrc"))),
      pfc_(consumes<reco::PFCandidateCollection>(params.getParameter<edm::InputTag>("pfSrc"))),
      pvsScore_(consumes<edm::ValueMap<float>>(params.getParameter<edm::InputTag>("pvSrc"))),
      goodPvCut_(params.getParameter<std::string>("goodPvCut"), true),
      goodPvCutString_(params.getParameter<std::string>("goodPvCut")),
      pvName_(params.getParameter<std::string>("pvName")),
      dlenMin_(params.getParameter<double>("dlenMin")),
      dlenSigMin_(params.getParameter<double>("dlenSigMin"))

{
  produces<nanoaod::FlatTable>("PV");
  produces<edm::PtrVector<reco::VertexCompositePtrCandidate>>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void HLTVertexTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //vertex collection
  auto pvsIn = iEvent.getHandle(pvs_);
  if (!pvsIn.isValid()) {
    edm::LogWarning("HLTVertexTableProducer")
        << "Invalid handle for " << pvName_ << " in primary vertex input collection";
    return;
  }
  const auto& pvsScoreProd = iEvent.get(pvsScore_);

  //pf candidates collection
  auto pfcIn = iEvent.getHandle(pfc_);
  if (!pfcIn.isValid()) {
    edm::LogWarning("HLTVertexTableProducer")
        << "Invalid handle for " << pvName_ << " in PF candidate input collection";
    return;
  }

  std::vector<float> v_ndof;
  std::vector<float> v_chi2;
  std::vector<float> v_x;
  std::vector<float> v_y;
  std::vector<float> v_z;
  std::vector<float> v_xError;
  std::vector<float> v_yError;
  std::vector<float> v_zError;
  std::vector<uint8_t> v_is_good;
  std::vector<uint8_t> v_nTracks;
  std::vector<float> v_pv_score;
  std::vector<float> v_pv_sumpt2;
  std::vector<float> v_pv_sumpx;
  std::vector<float> v_pv_sumpy;

  for (size_t i = 0; i < (*pvsIn).size(); i++) {
    v_ndof.push_back((*pvsIn)[i].ndof());
    v_chi2.push_back((*pvsIn)[i].normalizedChi2());
    v_x.push_back((*pvsIn)[i].x());
    v_y.push_back((*pvsIn)[i].y());
    v_z.push_back((*pvsIn)[i].z());
    v_xError.push_back((*pvsIn)[i].xError());
    v_yError.push_back((*pvsIn)[i].yError());
    v_zError.push_back((*pvsIn)[i].zError());
    v_nTracks.push_back((*pvsIn)[i].nTracks());
    v_is_good.push_back(goodPvCut_((*pvsIn)[i]));
    v_pv_score.push_back(pvsScoreProd.get(pvsIn.id(), i));

    float pv_sumpt2 = 0;
    float pv_sumpx = 0;
    float pv_sumpy = 0;
    for (const auto& obj : *pfcIn) {
      // skip neutrals
      if (obj.charge() == 0)
        continue;
      double dz = fabs(obj.trackRef()->dz((*pvsIn)[i].position()));
      bool include_pfc = false;
      if (dz < 0.2) {
        include_pfc = true;
        for (size_t j = 0; j < (*pvsIn).size() && j != i; j++) {
          double newdz = fabs(obj.trackRef()->dz((*pvsIn)[j].position()));
          if (newdz < dz) {
            include_pfc = false;
            break;
          }
        }  // this pf candidate belongs to other PV
      }
      if (include_pfc) {
        float pfc_pt = obj.pt();
        pv_sumpt2 += pfc_pt * pfc_pt;
        pv_sumpx += obj.px();
        pv_sumpy += obj.py();
      }
    }

    v_pv_sumpt2.push_back(pv_sumpt2);
    v_pv_sumpx.push_back(pv_sumpx);
    v_pv_sumpy.push_back(pv_sumpy);
  }

  //table for all primary vertices
  auto pvTable = std::make_unique<nanoaod::FlatTable>((*pvsIn).size(), pvName_, true);
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

  desc.add<std::string>("pvName")->setComment("name of the flat table ouput");
  desc.add<edm::InputTag>("pvSrc")->setComment(
      "std::vector<reco::Vertex> and ValueMap<float> primary vertex input collections");
  desc.add<edm::InputTag>("pfSrc")->setComment("reco::PFCandidateCollection PF candidates input collections");
  desc.add<std::string>("goodPvCut")->setComment("selection on the primary vertex");

  desc.add<double>("dlenMin")->setComment("minimum value of dl to select secondary vertex");
  desc.add<double>("dlenSigMin")->setComment("minimum value of dl significance to select secondary vertex");

  descriptions.addWithDefaultLabel(desc);
}

// ------------ define this as a plug-in ------------
DEFINE_FWK_MODULE(HLTVertexTableProducer);

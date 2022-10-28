#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorDRN.h"

#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/Transition.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaTools/interface/EgammaHGCALIDParamDefaults.h"

#include <vdt/vdtMath.h>

static const float RHO_MAX = 15.0f;
static const float X_MAX = 150.0f;
static const float X_RANGE = 300.0f;
static const float Y_MAX = 150.0f;
static const float Y_RANGE = 300.0f;
static const float Z_MAX = 330.0f;
static const float Z_RANGE = 660.0f;
static const float E_RANGE = 250.0f;

SCEnergyCorrectorDRN::SCEnergyCorrectorDRN() : caloTopo_(nullptr), caloGeom_(nullptr) {}

SCEnergyCorrectorDRN::SCEnergyCorrectorDRN(const edm::ParameterSet& iConfig, edm::ConsumesCollector cc)
    : SCEnergyCorrectorDRN() {
  setTokens(iConfig, cc);
}

void SCEnergyCorrectorDRN::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<edm::InputTag>("ecalRecHitsEE", edm::InputTag("ecalRecHit", "reducedEcalRecHitsEE"));
  desc.add<edm::InputTag>("ecalRecHitsEB", edm::InputTag("ecalRecHit", "reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("rhoFastJet", edm::InputTag("fixedGridRhoAll"));
}

edm::ParameterSetDescription SCEnergyCorrectorDRN::makePSetDescription() {
  edm::ParameterSetDescription desc;
  fillPSetDescription(desc);
  return desc;
}

void SCEnergyCorrectorDRN::setEventSetup(const edm::EventSetup& es) {
  caloTopo_ = &es.getData(caloTopoToken_);
  caloGeom_ = &es.getData(caloGeomToken_);
}

void SCEnergyCorrectorDRN::setEvent(const edm::Event& event) {
  event.getByToken(tokenEBRecHits_, recHitsEB_);
  event.getByToken(tokenEERecHits_, recHitsEE_);
  event.getByToken(rhoToken_, rhoHandle_);
}

void SCEnergyCorrectorDRN::makeInput(const edm::Event& iEvent,
                                     TritonInputMap& iInput,
                                     const reco::SuperClusterCollection& inputSCs) const {
  std::vector<unsigned> nHits;
  nHits.reserve(inputSCs.size());
  unsigned totalHits = 0;
  unsigned n;
  for (const auto& inputSC : inputSCs) {
    n = inputSC.hitsAndFractions().size();
    totalHits += n;
    nHits.push_back(n);
  }

  //set shapes
  auto& input1 = iInput.at("x__0");
  input1.setShape(0, totalHits);
  auto data1 = input1.allocate<float>();
  auto& vdata1 = (*data1)[0];

  auto& input2 = iInput.at("batch__1");
  input2.setShape(0, totalHits);
  auto data2 = input2.allocate<int64_t>();
  auto& vdata2 = (*data2)[0];

  auto& input3 = iInput.at("graphx__2");
  input3.setShape(0, 2 * nHits.size());
  auto data3 = input3.allocate<float>();
  auto& vdata3 = (*data3)[0];

  //fill
  unsigned batchNum = 0;
  float En, frac, x, y, z;
  for (const auto& inputSC : inputSCs) {
    const auto& hits = inputSC.hitsAndFractions();
    const bool isEB = hits[0].first.subdetId() == EcalBarrel;
    const auto& recHitsProduct = isEB ? recHitsEB_.product() : recHitsEE_.product();
    for (const auto& hit : hits) {
      En = EcalClusterTools::recHitEnergy(hit.first, recHitsProduct);
      frac = hit.second;
      GlobalPoint position = caloGeom_->getGeometry(hit.first)->getPosition();
      x = (position.x() + X_MAX) / X_RANGE;
      y = (position.y() + Y_MAX) / Y_RANGE;
      z = (position.z() + Z_MAX) / Z_RANGE;
      vdata1.push_back(x);
      vdata1.push_back(y);
      vdata1.push_back(z);
      vdata1.push_back(En * frac / E_RANGE);
      //Triton does not currently support batching for pytorch GNNs
      //We pass batch indices explicitely
      vdata2.push_back(batchNum);
    }
    vdata3.push_back(*rhoHandle_ / RHO_MAX);
    vdata3.push_back(0.0);
    ++batchNum;
  }

  // convert to server format
  input1.toServer(data1);
  input2.toServer(data2);
  input3.toServer(data3);
}

TritonOutput<float> SCEnergyCorrectorDRN::getOutput(const TritonOutputMap& iOutput) {
  //check the results
  const auto& output1 = iOutput.begin()->second;
  // convert from server format
  const auto& serverout = output1.fromServer<float>();

  return serverout;
}

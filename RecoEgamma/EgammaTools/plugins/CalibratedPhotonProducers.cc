//author: Alan Smithee
//description:
//  this class allows the residual scale and smearing to be applied to photon
//  it will write out all the calibration info in the event, such as scale correction value,
//  smearing correction value, random nr used, energy post calibration, energy pre calibration
//  can optionally write out a new collection of photon with the energy corrected by default
//  a port of EgammaAnalysis/ElectronTools/CalibratedPhotonProducerRun2

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "RecoEgamma/EgammaTools/interface/PhotonEnergyCalibrator.h"
#include "RecoEgamma/EgammaTools/interface/EGEnergySysIndex.h"
#include "RecoEgamma/EgammaTools/interface/EgammaRandomSeeds.h"

#include "TRandom2.h"

#include <memory>

#include <random>
#include <vector>

template <typename T>
class CalibratedPhotonProducerT : public edm::stream::EDProducer<> {
public:
  explicit CalibratedPhotonProducerT(const edm::ParameterSet&);
  ~CalibratedPhotonProducerT() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void setSemiDetRandomSeed(const edm::Event& iEvent, const T& obj, size_t nrObjs, size_t objNr);

  edm::EDGetTokenT<edm::View<T>> photonToken_;
  PhotonEnergyCalibrator energyCorrector_;
  std::unique_ptr<TRandom> semiDeterministicRng_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEBToken_;
  edm::EDGetTokenT<EcalRecHitCollection> recHitCollectionEEToken_;
  bool produceCalibratedObjs_;

  static const std::vector<int> valMapsToStore_;
};

template <typename T>
const std::vector<int> CalibratedPhotonProducerT<T>::valMapsToStore_ = {
    EGEnergySysIndex::kScaleStatUp,    EGEnergySysIndex::kScaleStatDown, EGEnergySysIndex::kScaleSystUp,
    EGEnergySysIndex::kScaleSystDown,  EGEnergySysIndex::kScaleGainUp,   EGEnergySysIndex::kScaleGainDown,
    EGEnergySysIndex::kSmearRhoUp,     EGEnergySysIndex::kSmearRhoDown,  EGEnergySysIndex::kSmearPhiUp,
    EGEnergySysIndex::kSmearPhiDown,   EGEnergySysIndex::kScaleUp,       EGEnergySysIndex::kScaleDown,
    EGEnergySysIndex::kSmearUp,        EGEnergySysIndex::kSmearDown,     EGEnergySysIndex::kScaleValue,
    EGEnergySysIndex::kSmearValue,     EGEnergySysIndex::kSmearNrSigma,  EGEnergySysIndex::kEcalPreCorr,
    EGEnergySysIndex::kEcalErrPreCorr, EGEnergySysIndex::kEcalPostCorr,  EGEnergySysIndex::kEcalErrPostCorr};

namespace {
  template <typename HandleType, typename ValType>
  void fillAndStoreValueMap(edm::Event& iEvent,
                            HandleType objHandle,
                            const std::vector<ValType>& vals,
                            const std::string& name) {
    auto valMap = std::make_unique<edm::ValueMap<ValType>>();
    typename edm::ValueMap<ValType>::Filler filler(*valMap);
    filler.insert(objHandle, vals.begin(), vals.end());
    filler.fill();
    iEvent.put(std::move(valMap), name);
  }
}  // namespace

template <typename T>
CalibratedPhotonProducerT<T>::CalibratedPhotonProducerT(const edm::ParameterSet& conf)
    : photonToken_(consumes(conf.getParameter<edm::InputTag>("src"))),
      energyCorrector_(conf.getParameter<std::string>("correctionFile")),
      recHitCollectionEBToken_(consumes(conf.getParameter<edm::InputTag>("recHitCollectionEB"))),
      recHitCollectionEEToken_(consumes(conf.getParameter<edm::InputTag>("recHitCollectionEE"))),
      produceCalibratedObjs_(conf.getParameter<bool>("produceCalibratedObjs")) {
  energyCorrector_.setMinEt(conf.getParameter<double>("minEtToCalibrate"));

  if (conf.getParameter<bool>("semiDeterministic")) {
    semiDeterministicRng_ = std::make_unique<TRandom2>();
    energyCorrector_.initPrivateRng(semiDeterministicRng_.get());
  }

  if (produceCalibratedObjs_)
    produces<std::vector<T>>();

  for (const auto& toStore : valMapsToStore_) {
    produces<edm::ValueMap<float>>(EGEnergySysIndex::name(toStore));
  }
}

template <typename T>
void CalibratedPhotonProducerT<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("recHitCollectionEB", edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("recHitCollectionEE", edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<std::string>("correctionFile", std::string());
  desc.add<double>("minEtToCalibrate", 5.0);
  desc.add<bool>("produceCalibratedObjs", true);
  desc.add<bool>("semiDeterministic", true);
  std::vector<std::string> valMapsProduced;
  valMapsProduced.reserve(valMapsToStore_.size());
  for (auto varToStore : valMapsToStore_)
    valMapsProduced.push_back(EGEnergySysIndex::name(varToStore));
  desc.add<std::vector<std::string>>("valueMapsStored", valMapsProduced)
      ->setComment(
          "provides to python configs the list of valuemaps stored, can not be overriden in the python config");
  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
void CalibratedPhotonProducerT<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto inHandle = iEvent.getHandle(photonToken_);

  auto recHitCollectionEBHandle = iEvent.getHandle(recHitCollectionEBToken_);
  auto recHitCollectionEEHandle = iEvent.getHandle(recHitCollectionEEToken_);

  std::unique_ptr<std::vector<T>> out = std::make_unique<std::vector<T>>();

  size_t nrObj = inHandle->size();
  std::array<std::vector<float>, EGEnergySysIndex::kNrSysErrs> results;
  for (auto& res : results)
    res.reserve(nrObj);

  const PhotonEnergyCalibrator::EventType evtType =
      iEvent.isRealData() ? PhotonEnergyCalibrator::EventType::DATA : PhotonEnergyCalibrator::EventType::MC;

  for (const auto& pho : *inHandle) {
    out->emplace_back(pho);

    if (semiDeterministicRng_)
      setSemiDetRandomSeed(iEvent, pho, nrObj, out->size());

    const EcalRecHitCollection* recHits =
        (pho.isEB()) ? recHitCollectionEBHandle.product() : recHitCollectionEEHandle.product();
    std::array<float, EGEnergySysIndex::kNrSysErrs> uncertainties =
        energyCorrector_.calibrate(out->back(), iEvent.id().run(), recHits, iEvent.streamID(), evtType);

    for (size_t index = 0; index < EGEnergySysIndex::kNrSysErrs; index++) {
      results[index].push_back(uncertainties[index]);
    }
  }

  auto fillAndStore = [&](auto handle) {
    for (const auto& mapToStore : valMapsToStore_) {
      fillAndStoreValueMap(iEvent, handle, results[mapToStore], EGEnergySysIndex::name(mapToStore));
    }
  };

  if (produceCalibratedObjs_) {
    fillAndStore(iEvent.put(std::move(out)));
  } else {
    fillAndStore(inHandle);
  }
}

//needs to be synced to CalibratedElectronProducers, want the same seed for a given SC
template <typename T>
void CalibratedPhotonProducerT<T>::setSemiDetRandomSeed(const edm::Event& iEvent,
                                                        const T& obj,
                                                        size_t nrObjs,
                                                        size_t objNr) {
  if (obj.superCluster().isNonnull()) {
    semiDeterministicRng_->SetSeed(egamma::getRandomSeedFromSC(iEvent, obj.superCluster()));
  } else {
    semiDeterministicRng_->SetSeed(egamma::getRandomSeedFromObj(iEvent, obj, nrObjs, objNr));
  }
}

using CalibratedPhotonProducer = CalibratedPhotonProducerT<reco::Photon>;
using CalibratedPatPhotonProducer = CalibratedPhotonProducerT<pat::Photon>;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CalibratedPhotonProducer);
DEFINE_FWK_MODULE(CalibratedPatPhotonProducer);

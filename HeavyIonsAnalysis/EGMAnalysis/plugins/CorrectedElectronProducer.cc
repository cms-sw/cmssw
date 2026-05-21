/*
 *  adapted from:
 *    RecoEgamma/EgammaTools/plugins/CalibratedElectronProducers.cc
 *    RecoEgamma/EgammaTools/src/ElectronEnergyCalibrator.cc
 *  extended + simplified for HI use
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "HeavyIonsAnalysis/EGMAnalysis/interface/EnergyScaleCorrector.h"
#include "RecoEgamma/EgammaTools/interface/EgammaRandomSeeds.h"
#include "RecoEgamma/EgammaTools/interface/EpCombinationTool.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "TRandom2.h"

#include <memory>
#include <vector>

template <typename T>
class CorrectedElectronProducerT : public edm::stream::EDProducer<> {
public:
  explicit CorrectedElectronProducerT(const edm::ParameterSet&);
  ~CorrectedElectronProducerT() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void setRandomSeed(const edm::Event& iEvent, const T& obj, size_t size, size_t index);

  bool semiDeterministic_;
  std::unique_ptr<TRandom2> semiDeterministicRng_;
  edm::EDGetTokenT<edm::View<T>> electronToken_;
  edm::EDGetTokenT<int> centralityToken_;
  EpCombinationTool epCombinator_;
  EnergyScaleCorrector energyCorrector_;
};

template <typename T>
CorrectedElectronProducerT<T>::CorrectedElectronProducerT(const edm::ParameterSet& conf)
    : semiDeterministic_(conf.getParameter<bool>("semiDeterministic")),
      semiDeterministicRng_(new TRandom2()),
      electronToken_(consumes<edm::View<T>>(conf.getParameter<edm::InputTag>("src"))),
      centralityToken_(consumes<int>(conf.getParameter<edm::InputTag>("centrality"))),
      epCombinator_{conf.getParameter<edm::ParameterSet>("epCombConfig"), consumesCollector()},
      energyCorrector_(conf.getParameter<std::string>("correctionFile"),
                       epCombinator_,
                       semiDeterministicRng_.get(),
                       conf.getParameter<double>("minPt")) {
  produces<std::vector<T>>();
}

template <typename T>
void CorrectedElectronProducerT<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("centrality", edm::InputTag("centralityBin"));
  desc.add<edm::ParameterSetDescription>("epCombConfig", EpCombinationTool::makePSetDescription());
  desc.add<std::string>("correctionFile", std::string());
  desc.add<double>("minPt", 20.0);
  desc.add<bool>("semiDeterministic", true);
  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
void CorrectedElectronProducerT<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  epCombinator_.setEventContent(iSetup);

  edm::Handle<edm::View<T>> electrons;
  iEvent.getByToken(electronToken_, electrons);
  edm::Handle<int> bin;
  iEvent.getByToken(centralityToken_, bin);

  auto out = std::make_unique<std::vector<T>>();
  for (const auto& ele : *electrons) {
    out->push_back(ele);
    auto pout = dynamic_cast<pat::Electron*>(&(out->back()));
    if (pout) {
      pout->addUserFloat("rawPt", ele.pt());
      pout->addUserFloat("rawEcalEnergy", ele.ecalEnergy());
    }

    if (semiDeterministic_)
      setRandomSeed(iEvent, ele, electrons->size(), out->size());

    energyCorrector_.calibrate(out->back(), *bin);
  }

  iEvent.put(std::move(out));
}

template <typename T>
void CorrectedElectronProducerT<T>::setRandomSeed(const edm::Event& iEvent, const T& obj, size_t size, size_t index) {
  semiDeterministicRng_->SetSeed(obj.superCluster().isNonnull()
                                     ? egamma::getRandomSeedFromSC(iEvent, obj.superCluster())
                                     : egamma::getRandomSeedFromObj(iEvent, obj, size, index));
}

using CorrectedElectronProducer = CorrectedElectronProducerT<reco::GsfElectron>;
using CorrectedPatElectronProducer = CorrectedElectronProducerT<pat::Electron>;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(CorrectedElectronProducer);
DEFINE_FWK_MODULE(CorrectedPatElectronProducer);

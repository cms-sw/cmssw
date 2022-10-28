#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonData.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "RecoEcal/EgammaClusterAlgos/interface/SCEnergyCorrectorDRN.h"

#include <sstream>
#include <string>
#include <vector>
#include <random>

/*
 * SCEnergyCorrectorDRNProducer
 *
 * Simple producer to generate a set of corrected superclusters with the DRN regression
 * Based on RecoEcal/EgammaClusterProducers/SCEnergyCorrectorProducer by S. Harper (RAL/CERN)
 *
 * Author: Simon Rothman (UMN, MIT)
 *
 */

namespace {
  float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

  float logcorrection(float x) {
    static float ln2 = log(2);
    return ln2 * 2 * (sigmoid(x) - 0.5);
  }

  float correction(float x) { return exp(-logcorrection(x)); }
}  // namespace

class SCEnergyCorrectorDRNProducer : public TritonEDProducer<> {
public:
  explicit SCEnergyCorrectorDRNProducer(const edm::ParameterSet& iConfig);

  void beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup) override;

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& input) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup, Output const& iOutput) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  SCEnergyCorrectorDRN energyCorrector_;
  edm::EDGetTokenT<reco::SuperClusterCollection> inputSCToken_;
};

SCEnergyCorrectorDRNProducer::SCEnergyCorrectorDRNProducer(const edm::ParameterSet& iConfig)
    : TritonEDProducer<>(iConfig),
      energyCorrector_(iConfig.getParameterSet("correctorCfg"), consumesCollector()),
      inputSCToken_(consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("inputSCs"))) {
  produces<reco::SuperClusterCollection>();
}

void SCEnergyCorrectorDRNProducer::beginLuminosityBlock(const edm::LuminosityBlock& iLumi,
                                                        const edm::EventSetup& iSetup) {
  energyCorrector_.setEventSetup(iSetup);
}

void SCEnergyCorrectorDRNProducer::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) {
  const auto& inputSCs = iEvent.get(inputSCToken_);

  if (inputSCs.empty()) {
    client_->setBatchSize(0);
    return;
  } else {
    client_->setBatchSize(1);
  }

  energyCorrector_.setEvent(iEvent);
  energyCorrector_.makeInput(iEvent, iInput, inputSCs);
}

void SCEnergyCorrectorDRNProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup, Output const& iOutput) {
  const auto& inputSCs = iEvent.get(inputSCToken_);
  if (inputSCs.empty())
    return;

  const auto& serverout = energyCorrector_.getOutput(iOutput);

  auto corrSCs = std::make_unique<reco::SuperClusterCollection>();
  unsigned i = 0;
  for (const auto& inputSC : inputSCs) {
    float corrEn = correction(serverout[0][0 + 6 * i]) * inputSC.rawEnergy();
    corrSCs->push_back(inputSC);
    corrSCs->back().setEnergy(corrEn);
    corrSCs->back().setCorrectedEnergy(corrEn);
    ++i;
  }

  auto scHandle = iEvent.put(std::move(corrSCs));
}

void SCEnergyCorrectorDRNProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::ParameterSetDescription>("correctorCfg", SCEnergyCorrectorDRN::makePSetDescription());
  TritonClient::fillPSetDescription(desc);
  desc.add<edm::InputTag>("inputSCs", edm::InputTag("particleFlowSuperClusterECAL"));
  descriptions.add("scEnergyCorrectorDRNProducer", desc);
}

DEFINE_FWK_MODULE(SCEnergyCorrectorDRNProducer);

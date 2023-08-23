/**\class SiStripClusterizerConditionsGPUESProducer
 *
 * Create a GPU cache object for fast access to conditions needed by the SiStrip clusterizer
 *
 * @see SiStripClusterizerConditions
 */
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoLocalTracker/Records/interface/SiStripClusterizerConditionsRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsGPU.h"

using namespace stripgpu;

class SiStripClusterizerConditionsGPUESProducer : public edm::ESProducer {
public:
  SiStripClusterizerConditionsGPUESProducer(const edm::ParameterSet&);
  ~SiStripClusterizerConditionsGPUESProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using ReturnType = std::unique_ptr<SiStripClusterizerConditionsGPU>;
  ReturnType produce(const SiStripClusterizerConditionsRcd&);

private:
  edm::ESGetToken<SiStripGain, SiStripGainRcd> gainToken_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualityToken_;
};

SiStripClusterizerConditionsGPUESProducer::SiStripClusterizerConditionsGPUESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("Label"));

  gainToken_ = cc.consumesFrom<SiStripGain, SiStripGainRcd>();
  noisesToken_ = cc.consumesFrom<SiStripNoises, SiStripNoisesRcd>();
  qualityToken_ = cc.consumesFrom<SiStripQuality, SiStripQualityRcd>(
      edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});
}

void SiStripClusterizerConditionsGPUESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("QualityLabel", "");
  desc.add<std::string>("Label", "");
  descriptions.add("SiStripClusterizerConditionsGPUESProducer", desc);
}

SiStripClusterizerConditionsGPUESProducer::ReturnType SiStripClusterizerConditionsGPUESProducer::produce(
    const SiStripClusterizerConditionsRcd& iRecord) {
  auto gainsH = iRecord.getTransientHandle(gainToken_);
  const auto& noises = iRecord.get(noisesToken_);
  const auto& quality = iRecord.get(qualityToken_);

  return std::make_unique<SiStripClusterizerConditionsGPU>(quality, gainsH.product(), noises);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripClusterizerConditionsGPUESProducer);

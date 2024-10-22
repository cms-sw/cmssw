/**\class SiStripClusterizerConditionsESProducer
 *
 * Create a cache object for fast access to conditions needed by the SiStrip clusterizer
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
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditions.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

class SiStripClusterizerConditionsESProducer : public edm::ESProducer {
public:
  SiStripClusterizerConditionsESProducer(const edm::ParameterSet&);
  ~SiStripClusterizerConditionsESProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  using ReturnType = std::unique_ptr<SiStripClusterizerConditions>;
  ReturnType produce(const SiStripClusterizerConditionsRcd&);

private:
  edm::ESGetToken<SiStripGain, SiStripGainRcd> m_gainToken;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> m_noisesToken;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> m_qualityToken;
};

SiStripClusterizerConditionsESProducer::SiStripClusterizerConditionsESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this, iConfig.getParameter<std::string>("Label"));

  m_gainToken = cc.consumesFrom<SiStripGain, SiStripGainRcd>();
  m_noisesToken = cc.consumesFrom<SiStripNoises, SiStripNoisesRcd>();
  m_qualityToken = cc.consumesFrom<SiStripQuality, SiStripQualityRcd>(
      edm::ESInputTag{"", iConfig.getParameter<std::string>("QualityLabel")});
}

void SiStripClusterizerConditionsESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("QualityLabel", "");
  desc.add<std::string>("Label", "");
  descriptions.add("SiStripClusterizerConditionsESProducer", desc);
}

SiStripClusterizerConditionsESProducer::ReturnType SiStripClusterizerConditionsESProducer::produce(
    const SiStripClusterizerConditionsRcd& iRecord) {
  auto gainsH = iRecord.getTransientHandle(m_gainToken);
  const auto& noises = iRecord.get(m_noisesToken);
  const auto& quality = iRecord.get(m_qualityToken);

  auto product = std::make_unique<SiStripClusterizerConditions>(&quality);

  const auto& connected = quality.cabling()->connected();
  const auto& detCabling = quality.cabling()->getDetCabling();
  product->reserve(connected.size());
  for (const auto& conn : connected) {
    const auto det = conn.first;
    if (!quality.IsModuleBad(det)) {
      const auto gainRange = gainsH->getRange(det);
      std::vector<float> invGains;
      invGains.reserve(6);
      std::transform(
          gainRange.first, gainRange.second, std::back_inserter(invGains), [](auto gain) { return 1.f / gain; });

      static const std::vector<const FedChannelConnection*> noConn{};
      const auto detConn_it = detCabling.find(det);

      product->emplace_back(det,
                            quality.getRange(det),
                            noises.getRange(det),
                            invGains,
                            (detCabling.end() != detConn_it) ? (*detConn_it).second : noConn);
    }
  }
  LogDebug("SiStripClusterizerConditionsESProducer")
      << "Produced a SiStripClusterizerConditions object for " << product->allDets().size() << " modules";
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripClusterizerConditionsESProducer);

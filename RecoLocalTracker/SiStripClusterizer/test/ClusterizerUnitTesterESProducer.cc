// system includes
#include <memory>

// user includes
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ClusterizerUnitTesterESProducer : public edm::ESProducer {
  typedef edm::ParameterSet PSet;
  typedef std::vector<PSet> VPSet;
  typedef VPSet::const_iterator iter_t;

public:
  ClusterizerUnitTesterESProducer(const PSet&);
  ~ClusterizerUnitTesterESProducer() {}
  std::shared_ptr<const SiStripGain> produceGainRcd(const SiStripGainRcd&) { return gain_; }
  std::shared_ptr<const SiStripNoises> produceNoisesRcd(const SiStripNoisesRcd&) { return noises_; }
  std::shared_ptr<const SiStripQuality> produceQualityRcd(const SiStripQualityRcd&) { return quality_; }

private:
  void extractNoiseGainQuality(const PSet&, SiStripQuality*, SiStripApvGain*, SiStripNoises*);
  void extractNoiseGainQualityForDetId(uint32_t, const VPSet&, SiStripQuality*, SiStripApvGain*, SiStripNoises*);

  void setNoises(uint32_t, std::vector<std::pair<uint16_t, float> >&, SiStripNoises*);
  void setGains(uint32_t, std::vector<std::pair<uint16_t, float> >&, SiStripApvGain*);

  // These objects might be shared across multiple concurrent
  // IOVs and are not allowed to be modified after the module
  // constructor finishes.
  std::unique_ptr<const SiStripApvGain> apvGain_;
  std::shared_ptr<const SiStripGain> gain_;
  std::shared_ptr<const SiStripNoises> noises_;
  std::shared_ptr<const SiStripQuality> quality_;
};

ClusterizerUnitTesterESProducer::ClusterizerUnitTesterESProducer(const edm::ParameterSet& conf) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());

  auto quality = std::make_unique<SiStripQuality>(detInfo);
  auto apvGain = std::make_unique<SiStripApvGain>();
  auto noises = std::make_unique<SiStripNoises>();

  extractNoiseGainQuality(conf, quality.get(), apvGain.get(), noises.get());

  gain_ = std::make_shared<const SiStripGain>(*apvGain, 1, detInfo);

  quality->cleanUp();
  quality->fillBadComponents();

  quality_ = std::move(quality);
  apvGain_ = std::move(apvGain);
  noises_ = std::move(noises);

  setWhatProduced(this, &ClusterizerUnitTesterESProducer::produceGainRcd);
  setWhatProduced(this, &ClusterizerUnitTesterESProducer::produceNoisesRcd);
  setWhatProduced(this, &ClusterizerUnitTesterESProducer::produceQualityRcd);
}

void ClusterizerUnitTesterESProducer::extractNoiseGainQuality(const edm::ParameterSet& conf,
                                                              SiStripQuality* quality,
                                                              SiStripApvGain* apvGain,
                                                              SiStripNoises* noises) {
  uint32_t detId = 0;
  VPSet groups = conf.getParameter<VPSet>("ClusterizerTestGroups");
  for (iter_t group = groups.begin(); group < groups.end(); group++) {
    VPSet tests = group->getParameter<VPSet>("Tests");
    for (iter_t test = tests.begin(); test < tests.end(); test++)
      extractNoiseGainQualityForDetId(detId++, test->getParameter<VPSet>("Digis"), quality, apvGain, noises);
  }
}

void ClusterizerUnitTesterESProducer::extractNoiseGainQualityForDetId(
    uint32_t detId, const VPSet& digiset, SiStripQuality* quality, SiStripApvGain* apvGain, SiStripNoises* noises) {
  std::vector<std::pair<uint16_t, float> > detNoises;
  std::vector<std::pair<uint16_t, float> > detGains;
  std::vector<unsigned> detBadStrips;
  for (iter_t digi = digiset.begin(); digi < digiset.end(); digi++) {
    uint16_t strip = digi->getParameter<unsigned>("Strip");
    if (digi->getParameter<unsigned>("ADC") != 0) {
      detNoises.push_back(std::make_pair(strip, digi->getParameter<double>("Noise")));
      detGains.push_back(std::make_pair(strip, digi->getParameter<double>("Gain")));
    }
    if (!digi->getParameter<bool>("Quality"))
      detBadStrips.push_back(quality->encode(strip, 1));
  }
  setNoises(detId, detNoises, noises);
  setGains(detId, detGains, apvGain);
  if (detBadStrips.size())
    quality->add(detId, std::make_pair(detBadStrips.begin(), detBadStrips.end()));
}

void ClusterizerUnitTesterESProducer::setNoises(uint32_t detId,
                                                std::vector<std::pair<uint16_t, float> >& digiNoises,
                                                SiStripNoises* noises) {
  std::sort(digiNoises.begin(), digiNoises.end());
  std::vector<float> detnoise;
  for (std::vector<std::pair<uint16_t, float> >::const_iterator digi = digiNoises.begin(); digi < digiNoises.end();
       ++digi) {
    detnoise.resize(digi->first, 1);  //pad with default noise 1
    detnoise.push_back(digi->second);
  }
  if (detnoise.size() > 768)
    throw cms::Exception("Faulty noise construction") << "No strip numbers greater than 767 please" << std::endl;
  detnoise.resize(768, 1.0);

  SiStripNoises::InputVector theSiStripVector;
  for (uint16_t strip = 0; strip < detnoise.size(); strip++) {
    noises->setData(detnoise.at(strip), theSiStripVector);
  }
  noises->put(detId, theSiStripVector);
}

void ClusterizerUnitTesterESProducer::setGains(uint32_t detId,
                                               std::vector<std::pair<uint16_t, float> >& digiGains,
                                               SiStripApvGain* apvGain) {
  std::sort(digiGains.begin(), digiGains.end());
  std::vector<float> detApvGains;
  for (std::vector<std::pair<uint16_t, float> >::const_iterator digi = digiGains.begin(); digi < digiGains.end();
       ++digi) {
    if (detApvGains.size() <= digi->first / 128) {
      detApvGains.push_back(digi->second);
    } else if (detApvGains.at(digi->first / 128) != digi->second) {
      throw cms::Exception("Faulty gain construction.") << "  Only one gain setting per APV please.\n";
    }
  }
  detApvGains.resize(6, 1.);

  SiStripApvGain::Range range(detApvGains.begin(), detApvGains.end());
  if (!apvGain->put(detId, range))
    throw cms::Exception("Trying to set gain twice for same detId: ") << detId;
  return;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
DEFINE_FWK_EVENTSETUP_MODULE(ClusterizerUnitTesterESProducer);

#include "CalibTracker/SiStripCommon/interface/ShallowClustersProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripProcessedRawDigi.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

ShallowClustersProducer::ShallowClustersProducer(const edm::ParameterSet& iConfig)
    : Prefix(iConfig.getParameter<std::string>("Prefix")), siStripClusterInfo_(consumesCollector()) {
  produces<std::vector<unsigned>>(Prefix + "number");
  produces<std::vector<unsigned>>(Prefix + "width");
  produces<std::vector<float>>(Prefix + "variance");
  produces<std::vector<float>>(Prefix + "barystrip");
  produces<std::vector<float>>(Prefix + "middlestrip");
  produces<std::vector<unsigned>>(Prefix + "charge");
  produces<std::vector<float>>(Prefix + "noise");
  produces<std::vector<float>>(Prefix + "ston");
  produces<std::vector<unsigned>>(Prefix + "seedstrip");
  produces<std::vector<unsigned>>(Prefix + "seedindex");
  produces<std::vector<unsigned>>(Prefix + "seedcharge");
  produces<std::vector<float>>(Prefix + "seednoise");
  produces<std::vector<float>>(Prefix + "seedgain");
  produces<std::vector<unsigned>>(Prefix + "qualityisbad");

  produces<std::vector<float>>(Prefix + "rawchargeC");
  produces<std::vector<float>>(Prefix + "rawchargeL");
  produces<std::vector<float>>(Prefix + "rawchargeR");
  produces<std::vector<float>>(Prefix + "rawchargeLL");
  produces<std::vector<float>>(Prefix + "rawchargeRR");
  produces<std::vector<float>>(Prefix + "eta");
  produces<std::vector<float>>(Prefix + "foldedeta");
  produces<std::vector<float>>(Prefix + "etaX");
  produces<std::vector<float>>(Prefix + "etaasymm");
  produces<std::vector<float>>(Prefix + "outsideasymm");
  produces<std::vector<float>>(Prefix + "neweta");
  produces<std::vector<float>>(Prefix + "newetaerr");

  produces<std::vector<unsigned>>(Prefix + "detid");
  produces<std::vector<int>>(Prefix + "subdetid");
  produces<std::vector<int>>(Prefix + "module");
  produces<std::vector<int>>(Prefix + "side");
  produces<std::vector<int>>(Prefix + "layerwheel");
  produces<std::vector<int>>(Prefix + "stringringrod");
  produces<std::vector<int>>(Prefix + "petal");
  produces<std::vector<int>>(Prefix + "stereo");

  theClustersToken_ = consumes<edmNew::DetSetVector<SiStripCluster>>(iConfig.getParameter<edm::InputTag>("Clusters"));
  theDigisToken_ = consumes<edm::DetSetVector<SiStripProcessedRawDigi>>(edm::InputTag("siStripProcessedRawDigis", ""));
}

void ShallowClustersProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  siStripClusterInfo_.initEvent(iSetup);

  auto number = std::make_unique<std::vector<unsigned>>(7, 0);
  auto width = std::make_unique<std::vector<unsigned>>();
  auto variance = std::make_unique<std::vector<float>>();
  auto barystrip = std::make_unique<std::vector<float>>();
  auto middlestrip = std::make_unique<std::vector<float>>();
  auto charge = std::make_unique<std::vector<unsigned>>();
  auto noise = std::make_unique<std::vector<float>>();
  auto ston = std::make_unique<std::vector<float>>();
  auto seedstrip = std::make_unique<std::vector<unsigned>>();
  auto seedindex = std::make_unique<std::vector<unsigned>>();
  auto seedcharge = std::make_unique<std::vector<unsigned>>();
  auto seednoise = std::make_unique<std::vector<float>>();
  auto seedgain = std::make_unique<std::vector<float>>();
  auto qualityisbad = std::make_unique<std::vector<unsigned>>();

  auto rawchargeC = std::make_unique<std::vector<float>>();
  auto rawchargeL = std::make_unique<std::vector<float>>();
  auto rawchargeR = std::make_unique<std::vector<float>>();
  auto rawchargeLL = std::make_unique<std::vector<float>>();
  auto rawchargeRR = std::make_unique<std::vector<float>>();
  auto etaX = std::make_unique<std::vector<float>>();
  auto eta = std::make_unique<std::vector<float>>();
  auto foldedeta = std::make_unique<std::vector<float>>();
  auto etaasymm = std::make_unique<std::vector<float>>();
  auto outsideasymm = std::make_unique<std::vector<float>>();
  auto neweta = std::make_unique<std::vector<float>>();
  auto newetaerr = std::make_unique<std::vector<float>>();

  auto detid = std::make_unique<std::vector<unsigned>>();
  auto subdetid = std::make_unique<std::vector<int>>();
  auto side = std::make_unique<std::vector<int>>();
  auto module = std::make_unique<std::vector<int>>();
  auto layerwheel = std::make_unique<std::vector<int>>();
  auto stringringrod = std::make_unique<std::vector<int>>();
  auto petal = std::make_unique<std::vector<int>>();
  auto stereo = std::make_unique<std::vector<int>>();

  edm::Handle<edmNew::DetSetVector<SiStripCluster>> clusters;
  //  iEvent.getByLabel(theClustersLabel, clusters);
  iEvent.getByToken(theClustersToken_, clusters);

  edm::Handle<edm::DetSetVector<SiStripProcessedRawDigi>> rawProcessedDigis;
  //  iEvent.getByLabel("siStripProcessedRawDigis", "", rawProcessedDigis);
  iEvent.getByToken(theDigisToken_, rawProcessedDigis);

  edmNew::DetSetVector<SiStripCluster>::const_iterator itClusters = clusters->begin();
  for (; itClusters != clusters->end(); ++itClusters) {
    uint32_t id = itClusters->id();
    const moduleVars moduleV(id, tTopo);
    for (edmNew::DetSet<SiStripCluster>::const_iterator cluster = itClusters->begin(); cluster != itClusters->end();
         ++cluster) {
      siStripClusterInfo_.setCluster(*cluster, id);
      const SiStripClusterInfo& info = siStripClusterInfo_;
      const NearDigis digis = rawProcessedDigis.isValid() ? NearDigis(info, *rawProcessedDigis) : NearDigis(info);

      (number->at(0))++;
      (number->at(moduleV.subdetid))++;
      width->push_back(cluster->amplitudes().size());
      barystrip->push_back(cluster->barycenter());
      variance->push_back(info.variance());
      middlestrip->push_back(info.firstStrip() + info.width() / 2.0);
      charge->push_back(info.charge());
      noise->push_back(info.noiseRescaledByGain());
      ston->push_back(info.signalOverNoise());
      seedstrip->push_back(info.maxStrip());
      seedindex->push_back(info.maxIndex());
      seedcharge->push_back(info.maxCharge());
      seednoise->push_back(info.stripNoisesRescaledByGain().at(info.maxIndex()));
      seedgain->push_back(info.stripGains().at(info.maxIndex()));
      qualityisbad->push_back(info.IsAnythingBad());

      rawchargeC->push_back(digis.max);
      rawchargeL->push_back(digis.left);
      rawchargeR->push_back(digis.right);
      rawchargeLL->push_back(digis.Lleft);
      rawchargeRR->push_back(digis.Rright);
      etaX->push_back(digis.etaX());
      eta->push_back(digis.eta());
      etaasymm->push_back(digis.etaasymm());
      outsideasymm->push_back(digis.outsideasymm());
      neweta->push_back((digis.last - digis.first) / info.charge());
      newetaerr->push_back((sqrt(digis.last + digis.first)) / pow(info.charge(), 1.5));

      detid->push_back(id);
      subdetid->push_back(moduleV.subdetid);
      side->push_back(moduleV.side);
      module->push_back(moduleV.module);
      layerwheel->push_back(moduleV.layerwheel);
      stringringrod->push_back(moduleV.stringringrod);
      petal->push_back(moduleV.petal);
      stereo->push_back(moduleV.stereo);
    }
  }

  iEvent.put(std::move(number), Prefix + "number");
  iEvent.put(std::move(width), Prefix + "width");
  iEvent.put(std::move(variance), Prefix + "variance");
  iEvent.put(std::move(barystrip), Prefix + "barystrip");
  iEvent.put(std::move(middlestrip), Prefix + "middlestrip");
  iEvent.put(std::move(charge), Prefix + "charge");
  iEvent.put(std::move(noise), Prefix + "noise");
  iEvent.put(std::move(ston), Prefix + "ston");
  iEvent.put(std::move(seedstrip), Prefix + "seedstrip");
  iEvent.put(std::move(seedindex), Prefix + "seedindex");
  iEvent.put(std::move(seedcharge), Prefix + "seedcharge");
  iEvent.put(std::move(seednoise), Prefix + "seednoise");
  iEvent.put(std::move(seedgain), Prefix + "seedgain");
  iEvent.put(std::move(qualityisbad), Prefix + "qualityisbad");

  iEvent.put(std::move(rawchargeC), Prefix + "rawchargeC");
  iEvent.put(std::move(rawchargeL), Prefix + "rawchargeL");
  iEvent.put(std::move(rawchargeR), Prefix + "rawchargeR");
  iEvent.put(std::move(rawchargeLL), Prefix + "rawchargeLL");
  iEvent.put(std::move(rawchargeRR), Prefix + "rawchargeRR");
  iEvent.put(std::move(etaX), Prefix + "etaX");
  iEvent.put(std::move(eta), Prefix + "eta");
  iEvent.put(std::move(foldedeta), Prefix + "foldedeta");
  iEvent.put(std::move(etaasymm), Prefix + "etaasymm");
  iEvent.put(std::move(outsideasymm), Prefix + "outsideasymm");
  iEvent.put(std::move(neweta), Prefix + "neweta");
  iEvent.put(std::move(newetaerr), Prefix + "newetaerr");

  iEvent.put(std::move(detid), Prefix + "detid");
  iEvent.put(std::move(subdetid), Prefix + "subdetid");
  iEvent.put(std::move(module), Prefix + "module");
  iEvent.put(std::move(side), Prefix + "side");
  iEvent.put(std::move(layerwheel), Prefix + "layerwheel");
  iEvent.put(std::move(stringringrod), Prefix + "stringringrod");
  iEvent.put(std::move(petal), Prefix + "petal");
  iEvent.put(std::move(stereo), Prefix + "stereo");
}

ShallowClustersProducer::NearDigis::NearDigis(const SiStripClusterInfo& info) {
  max = info.maxCharge();
  left = info.maxIndex() > uint16_t(0) ? info.stripCharges()[info.maxIndex() - 1] : 0;
  Lleft = info.maxIndex() > uint16_t(1) ? info.stripCharges()[info.maxIndex() - 2] : 0;
  right = unsigned(info.maxIndex() + 1) < info.stripCharges().size() ? info.stripCharges()[info.maxIndex() + 1] : 0;
  Rright = unsigned(info.maxIndex() + 2) < info.stripCharges().size() ? info.stripCharges()[info.maxIndex() + 2] : 0;
  first = info.stripCharges()[0];
  last = info.stripCharges()[info.width() - 1];
}

ShallowClustersProducer::NearDigis::NearDigis(const SiStripClusterInfo& info,
                                              const edm::DetSetVector<SiStripProcessedRawDigi>& rawProcessedDigis) {
  edm::DetSetVector<SiStripProcessedRawDigi>::const_iterator digiframe = rawProcessedDigis.find(info.detId());
  if (digiframe != rawProcessedDigis.end()) {
    max = digiframe->data.at(info.maxStrip()).adc();
    left = info.maxStrip() > uint16_t(0) ? digiframe->data.at(info.maxStrip() - 1).adc() : 0;
    Lleft = info.maxStrip() > uint16_t(1) ? digiframe->data.at(info.maxStrip() - 2).adc() : 0;
    right = unsigned(info.maxStrip() + 1) < digiframe->data.size() ? digiframe->data.at(info.maxStrip() + 1).adc() : 0;
    Rright = unsigned(info.maxStrip() + 2) < digiframe->data.size() ? digiframe->data.at(info.maxStrip() + 2).adc() : 0;
    first = digiframe->data.at(info.firstStrip()).adc();
    last = digiframe->data.at(info.firstStrip() + info.width() - 1).adc();
  } else {
    *this = NearDigis(info);
  }
}

ShallowClustersProducer::moduleVars::moduleVars(uint32_t detid, const TrackerTopology* tTopo) {
  SiStripDetId subdet(detid);
  subdetid = subdet.subDetector();
  if (SiStripDetId::TIB == subdetid) {
    module = tTopo->tibModule(detid);
    side = tTopo->tibIsZMinusSide(detid) ? -1 : 1;
    layerwheel = tTopo->tibLayer(detid);
    stringringrod = tTopo->tibString(detid);
    stereo = tTopo->tibIsStereo(detid) ? 1 : 0;
  } else if (SiStripDetId::TID == subdetid) {
    module = tTopo->tidModule(detid);
    side = tTopo->tidIsZMinusSide(detid) ? -1 : 1;
    layerwheel = tTopo->tidWheel(detid);
    stringringrod = tTopo->tidRing(detid);
    stereo = tTopo->tidIsStereo(detid) ? 1 : 0;
  } else if (SiStripDetId::TOB == subdetid) {
    module = tTopo->tobModule(detid);
    side = tTopo->tobIsZMinusSide(detid) ? -1 : 1;
    layerwheel = tTopo->tobLayer(detid);
    stringringrod = tTopo->tobRod(detid);
    stereo = tTopo->tobIsStereo(detid) ? 1 : 0;
  } else if (SiStripDetId::TEC == subdetid) {
    module = tTopo->tecModule(detid);
    side = tTopo->tecIsZMinusSide(detid) ? -1 : 1;
    layerwheel = tTopo->tecWheel(detid);
    stringringrod = tTopo->tecRing(detid);
    petal = tTopo->tecPetalNumber(detid);
    stereo = tTopo->tecIsStereo(detid) ? 1 : 0;
  } else {
    module = 0;
    side = 0;
    layerwheel = -1;
    stringringrod = -1;
    petal = -1;
  }
}

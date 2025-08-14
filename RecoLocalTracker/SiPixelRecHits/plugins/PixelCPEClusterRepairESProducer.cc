#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEClusterRepair.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

class PixelCPEClusterRepairESProducer : public edm::ESProducer {
public:
  PixelCPEClusterRepairESProducer(const edm::ParameterSet& p);
  ~PixelCPEClusterRepairESProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  std::unique_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord&);

private:
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> pDDToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> hTTToken_;
  edm::ESGetToken<SiPixelLorentzAngle, SiPixelLorentzAngleRcd> lorentzAngleToken_;
  edm::ESGetToken<std::vector<SiPixelTemplateStore>, SiPixelTemplateDBObjectESProducerRcd> templateStoreToken_;
  edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectESProducerRcd> templateDBobjectToken_;
  edm::ESGetToken<SiPixel2DTemplateDBObject, SiPixel2DTemplateDBObjectESProducerRcd> templateDBobject2DToken_;

  edm::ParameterSet pset_;
  bool doLorentzFromAlignment_;
  bool useLAFromDB_;
};

using namespace edm;

PixelCPEClusterRepairESProducer::PixelCPEClusterRepairESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");

  useLAFromDB_ = p.getParameter<bool>("useLAFromDB");
  doLorentzFromAlignment_ = p.getParameter<bool>("doLorentzFromAlignment");

  pset_ = p;
  auto c = setWhatProduced(this, myname);
  magfieldToken_ = c.consumes();
  pDDToken_ = c.consumes();
  hTTToken_ = c.consumes();
  templateStoreToken_ = c.consumes();
  templateDBobjectToken_ = c.consumes();
  templateDBobject2DToken_ = c.consumes();
  if (useLAFromDB_ || doLorentzFromAlignment_) {
    char const* laLabel = doLorentzFromAlignment_ ? "fromAlignment" : "";
    lorentzAngleToken_ = c.consumes(edm::ESInputTag("", laLabel));
  }
}

PixelCPEClusterRepairESProducer::~PixelCPEClusterRepairESProducer() {}

void PixelCPEClusterRepairESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // templates2
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "PixelCPEClusterRepair");

  // from PixelCPEBase
  PixelCPEBase::fillPSetDescription(desc);

  // from PixelCPEClusterRepair
  PixelCPEClusterRepair::fillPSetDescription(desc);

  // specific to PixelCPEClusterRepairESProducer
  descriptions.add("_templates2_default", desc);
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPEClusterRepairESProducer::produce(
    const TkPixelCPERecord& iRecord) {
  // Normal, default LA is used in case of template failure, load it unless
  // turned off
  // if turned off, null is ok, becomes zero
  const SiPixelLorentzAngle* lorentzAngleProduct = nullptr;
  if (useLAFromDB_ || doLorentzFromAlignment_) {
    lorentzAngleProduct = &iRecord.get(lorentzAngleToken_);
  }

  return std::make_unique<PixelCPEClusterRepair>(pset_,
                                                 &iRecord.get(magfieldToken_),
                                                 iRecord.get(pDDToken_),
                                                 iRecord.get(hTTToken_),
                                                 lorentzAngleProduct,
                                                 &iRecord.get(templateStoreToken_),
                                                 &iRecord.get(templateDBobjectToken_),
                                                 &iRecord.get(templateDBobject2DToken_));
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEClusterRepairESProducer);

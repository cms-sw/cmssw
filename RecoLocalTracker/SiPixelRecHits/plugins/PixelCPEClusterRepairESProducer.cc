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
  edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectESProducerRcd> templateDBobjectToken_;
  edm::ESGetToken<SiPixel2DTemplateDBObject, SiPixel2DTemplateDBObjectESProducerRcd> templateDBobject2DToken_;

  edm::ParameterSet pset_;
  bool DoLorentz_;
};

using namespace edm;

PixelCPEClusterRepairESProducer::PixelCPEClusterRepairESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");

  //DoLorentz_ = p.getParameter<bool>("DoLorentz"); // True when LA from alignment is used
  DoLorentz_ = p.getParameter<bool>("DoLorentz");

  pset_ = p;
  auto c = setWhatProduced(this, myname);
  c.setConsumes(magfieldToken_)
      .setConsumes(pDDToken_)
      .setConsumes(hTTToken_)
      .setConsumes(templateDBobjectToken_)
      .setConsumes(templateDBobject2DToken_);
  if (DoLorentz_) {
    c.setConsumes(lorentzAngleToken_, edm::ESInputTag("", "fromAlignment"));
  }

  //std::cout<<" from ES Producer Templates "<<myname<<" "<<DoLorentz_<<std::endl;  //dk
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
  desc.add<bool>("DoLorentz", true);
  descriptions.add("_templates2_default", desc);
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPEClusterRepairESProducer::produce(
    const TkPixelCPERecord& iRecord) {
  // Normal, default LA actually is NOT needed
  // null is ok becuse LA is not use by templates in this mode
  const SiPixelLorentzAngle* lorentzAngleProduct = nullptr;
  if (DoLorentz_) {  //  LA correction from alignment
    lorentzAngleProduct = &iRecord.get(lorentzAngleToken_);
  }

  return std::make_unique<PixelCPEClusterRepair>(pset_,
                                                 &iRecord.get(magfieldToken_),
                                                 iRecord.get(pDDToken_),
                                                 iRecord.get(hTTToken_),
                                                 lorentzAngleProduct,
                                                 &iRecord.get(templateDBobjectToken_),
                                                 &iRecord.get(templateDBobject2DToken_));
}

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEClusterRepairESProducer);

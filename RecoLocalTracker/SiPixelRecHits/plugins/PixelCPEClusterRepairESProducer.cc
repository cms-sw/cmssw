#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEClusterRepairESProducer.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEClusterRepair.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "CalibTracker/Records/interface/SiPixel2DTemplateDBObjectESProducerRcd.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <string>
#include <memory>

using namespace edm;

PixelCPEClusterRepairESProducer::PixelCPEClusterRepairESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");

  //DoLorentz_ = p.getParameter<bool>("DoLorentz"); // True when LA from alignment is used
  DoLorentz_ = p.existsAs<bool>("DoLorentz") ? p.getParameter<bool>("DoLorentz") : false;

  pset_ = p;
  setWhatProduced(this, myname);

  //std::cout<<" from ES Producer Templates "<<myname<<" "<<DoLorentz_<<std::endl;  //dk
}

PixelCPEClusterRepairESProducer::~PixelCPEClusterRepairESProducer() {}

void PixelCPEClusterRepairESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // templates2
  edm::ParameterSetDescription desc;
  desc.add<bool>("DoLorentz", true);
  desc.add<bool>("DoCosmics", false);
  desc.add<bool>("LoadTemplatesFromDB", true);
  desc.add<bool>("RunDamagedClusters", false);
  desc.add<std::string>("ComponentName", "PixelCPEClusterRepair");
  desc.add<double>("MinChargeRatio", 0.8);
  desc.add<double>("MaxSizeMismatchInY", 0.3);
  desc.add<bool>("Alpha2Order", true);
  desc.add<std::vector<std::string>>("Recommend2D",
                                     {
                                         "PXB 2",
                                         "PXB 3",
                                         "PXB 4",
                                     });
  desc.add<int>("ClusterProbComputationFlag", 0);
  desc.add<int>("speed", -2);
  desc.add<bool>("UseClusterSplitter", false);
  descriptions.add("templates2", desc);
}

std::unique_ptr<PixelClusterParameterEstimator> PixelCPEClusterRepairESProducer::produce(
    const TkPixelCPERecord& iRecord) {
  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield);

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(pDD);

  edm::ESHandle<TrackerTopology> hTT;
  iRecord.getRecord<TrackerDigiGeometryRecord>().getRecord<TrackerTopologyRcd>().get(hTT);

  edm::ESHandle<SiPixelLorentzAngle> lorentzAngle;
  const SiPixelLorentzAngle* lorentzAngleProduct = nullptr;
  if (DoLorentz_) {  //  LA correction from alignment
    iRecord.getRecord<SiPixelLorentzAngleRcd>().get("fromAlignment", lorentzAngle);
    lorentzAngleProduct = lorentzAngle.product();
  } else {  // Normal, deafult LA actually is NOT needed
    //iRecord.getRecord<SiPixelLorentzAngleRcd>().get(lorentzAngle);
    lorentzAngleProduct = nullptr;  // null is ok becuse LA is not use by templates in this mode
  }

  ESHandle<SiPixelTemplateDBObject> templateDBobject;
  iRecord.getRecord<SiPixelTemplateDBObjectESProducerRcd>().get(templateDBobject);

  ESHandle<SiPixel2DTemplateDBObject> templateDBobject2D;
  iRecord.getRecord<SiPixel2DTemplateDBObjectESProducerRcd>().get(templateDBobject2D);

  return std::make_unique<PixelCPEClusterRepair>(pset_,
                                                 magfield.product(),
                                                 *pDD.product(),
                                                 *hTT.product(),
                                                 lorentzAngleProduct,
                                                 templateDBobject.product(),
                                                 templateDBobject2D.product());
}

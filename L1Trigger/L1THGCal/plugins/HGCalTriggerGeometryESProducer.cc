
#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class HGCalTriggerGeometryESProducer : public edm::ESProducer {
public:
  HGCalTriggerGeometryESProducer(const edm::ParameterSet&);
  ~HGCalTriggerGeometryESProducer() override;

  typedef std::unique_ptr<HGCalTriggerGeometryBase> ReturnType;

  ReturnType produce(const CaloGeometryRecord&);

private:
  edm::ParameterSet geometry_config_;
  std::string geometry_name_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> ee_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> hsi_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> hsc_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> nose_geometry_token_;
  bool isV9Geometry_;
};

HGCalTriggerGeometryESProducer::HGCalTriggerGeometryESProducer(const edm::ParameterSet& iConfig)
    : geometry_config_(iConfig.getParameterSet("TriggerGeometry")),
      geometry_name_(geometry_config_.getParameter<std::string>("TriggerGeometryName")) {
  auto cc = setWhatProduced(this);
  ee_geometry_token_ = cc.consumes(edm::ESInputTag{"", "HGCalEESensitive"});
  hsi_geometry_token_ = cc.consumes(edm::ESInputTag{"", "HGCalHESiliconSensitive"});
  hsc_geometry_token_ = cc.consumes(edm::ESInputTag{"", "HGCalHEScintillatorSensitive"});
  nose_geometry_token_ = cc.consumes(edm::ESInputTag{"", "HGCalHFNoseSensitive"});
}

HGCalTriggerGeometryESProducer::~HGCalTriggerGeometryESProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

HGCalTriggerGeometryESProducer::ReturnType HGCalTriggerGeometryESProducer::produce(const CaloGeometryRecord& iRecord) {
  ReturnType geometry(HGCalTriggerGeometryFactory::get()->create(geometry_name_, geometry_config_));

  // Initialization with or without nose geometry
  if (iRecord.getHandle(nose_geometry_token_)) {
    geometry->setWithNoseGeometry(true);
    geometry->initialize(&iRecord.get(ee_geometry_token_),
                         &iRecord.get(hsi_geometry_token_),
                         &iRecord.get(hsc_geometry_token_),
                         &iRecord.get(nose_geometry_token_));
  } else {
    geometry->initialize(
        &iRecord.get(ee_geometry_token_), &iRecord.get(hsi_geometry_token_), &iRecord.get(hsc_geometry_token_));
  }
  return geometry;
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalTriggerGeometryESProducer);

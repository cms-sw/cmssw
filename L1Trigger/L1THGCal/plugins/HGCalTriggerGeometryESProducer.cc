
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
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> calo_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> ee_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> hsi_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> hsc_geometry_token_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> nose_geometry_token_;
  bool isV9Geometry_;
};

HGCalTriggerGeometryESProducer::HGCalTriggerGeometryESProducer(const edm::ParameterSet& iConfig)
    : geometry_config_(iConfig.getParameterSet("TriggerGeometry")),
      geometry_name_(geometry_config_.getParameter<std::string>("TriggerGeometryName")),
      isV9Geometry_(iConfig.getParameter<bool>("isV9Geometry")) {
  auto cc = setWhatProduced(this);
  if (isV9Geometry_) {
    cc.setConsumes(ee_geometry_token_, edm::ESInputTag{"", "HGCalEESensitive"})
        .setConsumes(hsi_geometry_token_, edm::ESInputTag{"", "HGCalHESiliconSensitive"})
        .setConsumes(hsc_geometry_token_, edm::ESInputTag{"", "HGCalHEScintillatorSensitive"})
        .setConsumes(nose_geometry_token_, edm::ESInputTag{"", "HGCalHFNoseSensitive"});
  } else {
    cc.setConsumes(calo_geometry_token_);
  }
}

HGCalTriggerGeometryESProducer::~HGCalTriggerGeometryESProducer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

HGCalTriggerGeometryESProducer::ReturnType HGCalTriggerGeometryESProducer::produce(const CaloGeometryRecord& iRecord) {
  // using namespace edm::es;
  ReturnType geometry(HGCalTriggerGeometryFactory::get()->create(geometry_name_, geometry_config_));
  if (isV9Geometry_) {
    // Initialize trigger geometry for V9 HGCAL geometry

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

  } else {
    // Initialize trigger geometry for V7/V8 HGCAL geometry
    const auto& calo_geometry = iRecord.get(calo_geometry_token_);
    if (not(calo_geometry.getSubdetectorGeometry(DetId::Forward, HGCEE) &&
            calo_geometry.getSubdetectorGeometry(DetId::Forward, HGCHEF) &&
            calo_geometry.getSubdetectorGeometry(DetId::Hcal, HcalEndcap))) {
      throw cms::Exception("LogicError")
          << "Configuration asked for non-V9 geometry, but the CaloGeometry does not look like one";
    }
    geometry->initialize(&calo_geometry);
  }
  return geometry;
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalTriggerGeometryESProducer);

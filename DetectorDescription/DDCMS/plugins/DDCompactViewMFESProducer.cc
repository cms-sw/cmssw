// -*- C++ -*-
//
// Package:    DetectorDescription/Core
// Class:      DDCompactViewMFESProducer
//
/**\class DDCompactViewMFESProducer

 Description: Produce DDCompactView

 Implementation:
     Allow users view a DDDetector as a legacy compact view
*/
//
// Original Author:  Ianna Osborne
//
//

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DD4hep/Detector.h"

using namespace std;
using namespace cms;

class DDCompactViewMFESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  DDCompactViewMFESProducer(const edm::ParameterSet&);
  ~DDCompactViewMFESProducer() override;

  using ReturnType = unique_ptr<DDCompactView>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const IdealMagneticFieldRecord&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
		      const edm::IOVSyncValue &,
		      edm::ValidityInterval &) override;
private:
  const string m_label;
};

DDCompactViewMFESProducer::DDCompactViewMFESProducer(const edm::ParameterSet& iConfig)
    : m_label(iConfig.getParameter<std::string>("appendToDataLabel")) {
  setWhatProduced(this);
  findingRecord<IdealMagneticFieldRecord>();
}

DDCompactViewMFESProducer::~DDCompactViewMFESProducer() {}

void DDCompactViewMFESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDCompactViewMFESProducer::ReturnType DDCompactViewMFESProducer::produce(const IdealMagneticFieldRecord& iRecord) {
  edm::ESHandle<DDDetector> det;
  iRecord.get(m_label, det);

  auto product = std::make_unique<DDCompactView>(*det);
  return product;
}

void
DDCompactViewMFESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
					const edm::IOVSyncValue &iosv,
					edm::ValidityInterval &oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDCompactViewMFESProducer);

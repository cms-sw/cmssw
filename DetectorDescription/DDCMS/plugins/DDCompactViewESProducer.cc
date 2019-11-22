// -*- C++ -*-
//
// Package:    DetectorDescription/Core
// Class:      DDCompactViewESProducer
//
/**\class DDCompactViewESProducer

 Description: Produce DDCompactView

 Implementation:
     Allow users view a DDDetector as a legacy compact view
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 22 May 2019 14:32:49 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

using namespace std;
using namespace cms;

class DDCompactViewESProducer : public edm::ESProducer {
public:
  DDCompactViewESProducer(const edm::ParameterSet&);
  ~DDCompactViewESProducer() override;

  using ReturnType = unique_ptr<DDCompactView>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const IdealGeometryRecord&);

private:
  const edm::ESGetToken<DDDetector, IdealGeometryRecord> m_detToken;
};

DDCompactViewESProducer::DDCompactViewESProducer(const edm::ParameterSet& iConfig)
    : m_detToken(setWhatProduced(this).consumes<DDDetector>(
          edm::ESInputTag("", iConfig.getParameter<std::string>("appendToDataLabel")))) {}

DDCompactViewESProducer::~DDCompactViewESProducer() {}

void DDCompactViewESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDCompactViewESProducer::ReturnType DDCompactViewESProducer::produce(const IdealGeometryRecord& iRecord) {
  auto product = std::make_unique<DDCompactView>(iRecord.get(m_detToken));
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDCompactViewESProducer);

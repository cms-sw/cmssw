// -*- C++ -*-
//
// Package:    DetectorDescription/DDDetectorESProducer
// Class:      DDDetectorESProducer
//
/**\class DDDetectorESProducer

 Description: Produce Detector description

 Implementation:
     Detector is described in XML
*/
//
// Original Author:  Ianna Osborne
//         Created:  Fri, 07 Dec 2018 11:20:52 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

using namespace std;
using namespace cms;
using namespace edm;

class DDDetectorESProducer : public ESProducer, public EventSetupRecordIntervalFinder {
public:
  DDDetectorESProducer(const ParameterSet&);
  ~DDDetectorESProducer() override;

  using ReturnType = unique_ptr<DDDetector>;
  using Detector = dd4hep::Detector;

  ReturnType produce(const GeometryFileRcd&);
  static void fillDescriptions(ConfigurationDescriptions&);

protected:
  void setIntervalFor(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&, ValidityInterval&) override;

private:
  const string m_confGeomXMLFiles;
  const string m_label;
};

DDDetectorESProducer::DDDetectorESProducer(const ParameterSet& iConfig)
    : m_confGeomXMLFiles(iConfig.getParameter<FileInPath>("confGeomXMLFiles").fullPath()),
      m_label(iConfig.getParameter<string>("appendToDataLabel")) {
  setWhatProduced(this);
  findingRecord<GeometryFileRcd>();
}

DDDetectorESProducer::~DDDetectorESProducer() {}

void DDDetectorESProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;

  desc.add<FileInPath>("confGeomXMLFiles");
  descriptions.addDefault(desc);
}

void DDDetectorESProducer::setIntervalFor(const eventsetup::EventSetupRecordKey& iKey,
                                          const IOVSyncValue& iTime,
                                          ValidityInterval& oInterval) {
  oInterval = ValidityInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());  //infinite
}

DDDetectorESProducer::ReturnType DDDetectorESProducer::produce(const GeometryFileRcd& iRecord) {
  LogDebug("Geometry") << "DDDetectorESProducer::Produce " << m_label;
  return make_unique<DDDetector>(m_label, m_confGeomXMLFiles);
}

DEFINE_FWK_EVENTSETUP_SOURCE(DDDetectorESProducer);

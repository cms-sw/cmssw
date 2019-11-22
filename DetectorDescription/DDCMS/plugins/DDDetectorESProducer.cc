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
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/MFGeometryFileRcd.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
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

  ReturnType produceGeom(const IdealGeometryRecord&);
  ReturnType produceMagField(const IdealMagneticFieldRecord&);
  ReturnType produce();
  static void fillDescriptions(ConfigurationDescriptions&);

protected:
  void setIntervalFor(const eventsetup::EventSetupRecordKey&, const IOVSyncValue&, ValidityInterval&) override;

private:
  const bool fromDB_;
  const string appendToDataLabel_;
  const string confGeomXMLFiles_;
  const string rootDDName_;
  const string label_;
  edm::ESGetToken<FileBlob, MFGeometryFileRcd> mfToken_;
  edm::ESGetToken<FileBlob, GeometryFileRcd> geomToken_;
};

DDDetectorESProducer::DDDetectorESProducer(const ParameterSet& iConfig)
    : fromDB_(iConfig.getParameter<bool>("fromDB")),
      appendToDataLabel_(iConfig.getParameter<string>("appendToDataLabel")),
      confGeomXMLFiles_(iConfig.getParameter<FileInPath>("confGeomXMLFiles").fullPath()),
      rootDDName_(iConfig.getParameter<string>("rootDDName")),
      label_(iConfig.getParameter<string>("label")) {
  if (rootDDName_ == "MagneticFieldVolumes:MAGF" || rootDDName_ == "cmsMagneticField:MAGF") {
    auto c = setWhatProduced(this,
                             &DDDetectorESProducer::produceMagField,
                             edm::es::Label(iConfig.getParameter<std::string>("@module_label")));
    if (fromDB_) {
      c.setConsumes(mfToken_, edm::ESInputTag("", label_));
    }
    findingRecord<IdealMagneticFieldRecord>();
  } else {
    auto c = setWhatProduced(
        this, &DDDetectorESProducer::produceGeom, edm::es::Label(iConfig.getParameter<std::string>("@module_label")));
    if (fromDB_) {
      c.setConsumes(geomToken_, edm::ESInputTag("", label_));
    }
    findingRecord<IdealGeometryRecord>();
  }
}

DDDetectorESProducer::~DDDetectorESProducer() {}

void DDDetectorESProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;

  desc.addOptional<FileInPath>("confGeomXMLFiles");
  desc.add<string>("rootDDName", "cms:OCMS");
  desc.add<string>("label", "");
  desc.add<bool>("fromDB", false);
  descriptions.add("DDDetectorESProducer", desc);

  edm::ParameterSetDescription descDB;
  descDB.add<string>("rootDDName", "cms:OCMS");
  descDB.add<string>("label", "Extended");
  descDB.add<bool>("fromDB", true);
  descriptions.add("DDDetectorESProducerFromDB", descDB);
}

void DDDetectorESProducer::setIntervalFor(const eventsetup::EventSetupRecordKey& iKey,
                                          const IOVSyncValue& iTime,
                                          ValidityInterval& oInterval) {
  oInterval = ValidityInterval(IOVSyncValue::beginOfTime(), IOVSyncValue::endOfTime());  //infinite
}

DDDetectorESProducer::ReturnType DDDetectorESProducer::produceMagField(const IdealMagneticFieldRecord& iRecord) {
  LogVerbatim("Geometry") << "DDDetectorESProducer::Produce MF " << appendToDataLabel_;
  if (fromDB_) {
    edm::ESTransientHandle<FileBlob> gdd = iRecord.getTransientHandle(mfToken_);
    unique_ptr<vector<unsigned char> > tb = (*gdd).getUncompressedBlob();

    return make_unique<cms::DDDetector>(label_, string(tb->begin(), tb->end()), true);
  } else {
    return make_unique<DDDetector>(appendToDataLabel_, confGeomXMLFiles_);
  }
}

DDDetectorESProducer::ReturnType DDDetectorESProducer::produceGeom(const IdealGeometryRecord& iRecord) {
  LogVerbatim("Geometry") << "DDDetectorESProducer::Produce " << appendToDataLabel_;
  if (fromDB_) {
    edm::ESTransientHandle<FileBlob> gdd = iRecord.getTransientHandle(geomToken_);
    unique_ptr<vector<unsigned char> > tb = (*gdd).getUncompressedBlob();

    return make_unique<cms::DDDetector>(label_, string(tb->begin(), tb->end()), true);
  } else {
    return make_unique<DDDetector>(appendToDataLabel_, confGeomXMLFiles_);
  }
}

DDDetectorESProducer::ReturnType DDDetectorESProducer::produce() {
  LogVerbatim("Geometry") << "DDDetectorESProducer::Produce " << appendToDataLabel_;
  return make_unique<DDDetector>(appendToDataLabel_, confGeomXMLFiles_);
}

DEFINE_FWK_EVENTSETUP_SOURCE(DDDetectorESProducer);

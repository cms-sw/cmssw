#ifndef GeometryReaders_XMLIdealGeometryESSource_XMLIdealGeometryESSource_H
#define GeometryReaders_XMLIdealGeometryESSource_XMLIdealGeometryESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordInfiniteIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <memory>
#include <string>

class XMLIdealGeometryESSource : public edm::ESProducer, public edm::EventSetupRecordInfiniteIntervalFinder {
public:
  XMLIdealGeometryESSource(const edm::ParameterSet &p);
  XMLIdealGeometryESSource(const XMLIdealGeometryESSource &) = delete;
  const XMLIdealGeometryESSource &operator=(const XMLIdealGeometryESSource &) = delete;
  ~XMLIdealGeometryESSource() override;
  std::unique_ptr<DDCompactView> produceGeom(const IdealGeometryRecord &);
  std::unique_ptr<DDCompactView> produceMagField(const IdealMagneticFieldRecord &);
  std::unique_ptr<DDCompactView> produce();

private:
  std::string rootNodeName_;
  bool userNS_;
  GeometryConfiguration geoConfig_;
};

#endif

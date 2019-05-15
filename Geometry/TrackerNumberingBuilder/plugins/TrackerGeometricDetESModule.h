#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
}

class GeometricDet;
class IdealGeometryRecord;

class TrackerGeometricDetESModule : public edm::ESProducer
{
public:
  TrackerGeometricDetESModule( const edm::ParameterSet & p );
  ~TrackerGeometricDetESModule( void ) override; 
  std::unique_ptr<GeometricDet> produce( const IdealGeometryRecord & );

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
  
private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<PGeometricDet, IdealGeometryRecord> pgToken_;
  bool fromDDD_;
};

#endif





#ifndef Geometry_MTDNumberingBuilder_MTDGeometricTimingDetESModule_H
#define Geometry_MTDNumberingBuilder_MTDGeometricTimingDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"

namespace edm {
  class ConfigurationDescriptions;
  class ParameterSet;
}

class GeometricTimingDet;
class IdealGeometryRecord;

class MTDGeometricTimingDetESModule : public edm::ESProducer
{
public:
  MTDGeometricTimingDetESModule( const edm::ParameterSet & p );
  ~MTDGeometricTimingDetESModule( void ) override; 
  std::unique_ptr<GeometricTimingDet> produce( const IdealGeometryRecord & );

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
  
private:
  bool fromDDD_;
};

#endif





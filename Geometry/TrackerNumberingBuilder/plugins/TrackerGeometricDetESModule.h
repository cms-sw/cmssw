#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"

namespace edm {
  class ConfigurationDescriptions;
}

class  TrackerGeometricDetESModule: public edm::ESProducer
{
public:
  TrackerGeometricDetESModule( const edm::ParameterSet & p );
  virtual ~TrackerGeometricDetESModule( void ); 
  std::auto_ptr<GeometricDet>       produce( const IdealGeometryRecord & );

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
private:
  bool fromDDD_;
  unsigned int layerNumberPXB_; // default 16; 18 for SLHC
  unsigned int totalBlade_;     // default 24; 56 for SLHC
};


#endif





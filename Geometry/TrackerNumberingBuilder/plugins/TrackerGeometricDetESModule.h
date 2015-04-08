#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"

#include <vector>

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
  std::vector<int> detidShifts_; // default 16; 18 for SLHC
};


#endif





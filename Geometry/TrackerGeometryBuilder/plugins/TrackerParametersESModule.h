#ifndef Geometry_TrackerGeometryBuilder_TrackerParametersESModule_H
#define Geometry_TrackerGeometryBuilder_TrackerParametersESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include <memory>

namespace edm {
  class ConfigurationDescriptions;
}
class PTrackerParameters;
class PTrackerParametersRcd;

class  TrackerParametersESModule: public edm::ESProducer
{
 public:
  TrackerParametersESModule( const edm::ParameterSet & );
  ~TrackerParametersESModule( void ) override;

  typedef std::shared_ptr<PTrackerParameters> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & );
  
  ReturnType produce( const PTrackerParametersRcd & );
};

#endif

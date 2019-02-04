#ifndef GEOMETRY_MTDNUMBERINGBUILDER_MTDTOPOLOGYEP_H
#define GEOMETRY_MTDNUMBERINGBUILDER_MTDTOPOLOGYEP_H 1

#include "memory"
#include "FWCore/Framework/interface/ESProducer.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"

namespace edm {
  class ConfigurationDescriptions;
}

class MTDTopologyEP : public edm::ESProducer
{
public:
  MTDTopologyEP( const edm::ParameterSet & );
  ~MTDTopologyEP( void ) override;

  using ReturnType = std::unique_ptr<MTDTopology>;

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
    
  ReturnType produce( const MTDTopologyRcd & );

private:
  void fillParameters( const PMTDParameters& );

  int mtdTopologyMode_;
  MTDTopology::BTLValues btlVals_;
  MTDTopology::ETLValues etlVals_;
};

#endif

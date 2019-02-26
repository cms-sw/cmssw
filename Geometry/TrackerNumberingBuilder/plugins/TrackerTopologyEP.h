#ifndef GEOMETRY_TRACKERNUMBERINGBUILDER_TRACKERTOPOLOGYEP_H
#define GEOMETRY_TRACKERNUMBERINGBUILDER_TRACKERTOPOLOGYEP_H 1

#include "memory"
#include "FWCore/Framework/interface/ESProducer.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "CondFormats/GeometryObjects/interface/PTrackerParameters.h"

namespace edm {
  class ConfigurationDescriptions;
}

class TrackerTopologyEP : public edm::ESProducer
{
public:
  TrackerTopologyEP( const edm::ParameterSet & );
  ~TrackerTopologyEP( void ) override;

  using ReturnType = std::unique_ptr<TrackerTopology>;

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
    
  ReturnType produce( const TrackerTopologyRcd & );

private:
  void fillParameters( const PTrackerParameters& );
    
  TrackerTopology::PixelBarrelValues pxbVals_;
  TrackerTopology::PixelEndcapValues pxfVals_;
  TrackerTopology::TECValues tecVals_;
  TrackerTopology::TIBValues tibVals_;
  TrackerTopology::TIDValues tidVals_;
  TrackerTopology::TOBValues tobVals_;
};

#endif

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

  using ReturnType = std::unique_ptr<TrackerTopology>;

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
    
  ReturnType produce( const TrackerTopologyRcd & );

private:
  void fillParameters( const PTrackerParameters&,
                       TrackerTopology::PixelBarrelValues& pxbVals,
                       TrackerTopology::PixelEndcapValues& pxfVals,
                       TrackerTopology::TECValues& tecVals,
                       TrackerTopology::TIBValues& tibVals,
                       TrackerTopology::TIDValues& tidVals,
                       TrackerTopology::TOBValues& tobVals);

  const edm::ESGetToken<PTrackerParameters, PTrackerParametersRcd> token_;
    
};

#endif

#ifndef GEOMETRY_TRACKERNUMBERINGBUILDER_TRACKERTOPOLOGYEP_H
#define GEOMETRY_TRACKERNUMBERINGBUILDER_TRACKERTOPOLOGYEP_H 1


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class TrackerTopologyEP : public edm::ESProducer {

public:
  TrackerTopologyEP(const edm::ParameterSet&);
  ~TrackerTopologyEP();

  typedef boost::shared_ptr<TrackerTopology> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & descriptions );
    
  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  TrackerTopology::PixelBarrelValues pxbVals_;
  TrackerTopology::PixelEndcapValues pxfVals_;
  TrackerTopology::TECValues tecVals_;
  TrackerTopology::TIBValues tibVals_;
  TrackerTopology::TIDValues tidVals_;
  TrackerTopology::TOBValues tobVals_;
  



};
#endif

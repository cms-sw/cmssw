
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef STACKED_TRACKER_GEOMETRY_ESMODULE_H
#define STACKED_TRACKER_GEOMETRY_ESMODULE_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"

#include <boost/shared_ptr.hpp>

#include <memory>
#include <string>

class  StackedTrackerGeometryESModule: public edm::ESProducer{
 public:
  StackedTrackerGeometryESModule(const edm::ParameterSet & p);
  virtual ~StackedTrackerGeometryESModule(); 
  boost::shared_ptr<StackedTrackerGeometry> produce(const StackedTrackerGeometryRecord & record);
 
 private:
  boost::shared_ptr <StackedTrackerGeometry> _tracker;

  double radial_window, phi_window, z_window;
  unsigned int truncation_precision;
  bool makeDebugFile;

};

//}

#endif

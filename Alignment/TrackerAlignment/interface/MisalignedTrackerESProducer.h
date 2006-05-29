#ifndef Alignment_MisalignedTrackerESProducer_MisalignedTrackerESProducerESProducer_h
#define Alignment_MisalignedTrackerESProducer_MisalignedTrackerESProducerESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <boost/shared_ptr.hpp>

///
/// An ESProducer that fills the TrackerDigiGeometryRcd with a misaligned tracker
/// 
/// This should replace the standard TrackerDigiGeometryESModule when producing
/// Misalignment scenarios.
/// FIXME: configuration file, output POOL-ORA object?
///
class MisalignedTrackerESProducer: public edm::ESProducer
{
public:

  /// Constructor
  MisalignedTrackerESProducer( const edm::ParameterSet & p );
  
  /// Destructor
  virtual ~MisalignedTrackerESProducer(); 
  
  /// Produce the misaligned tracker geometry and store it
  boost::shared_ptr<TrackerGeometry> produce( const TrackerDigiGeometryRecord& );

private:

  edm::ParameterSet theParameterSet;
  
  boost::shared_ptr<TrackerGeometry> theTracker;

};


#endif





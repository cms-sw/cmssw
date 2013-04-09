/// ////////////////////////////////////////
/// Stacked Tracker Simulations          ///
/// Written by:                          ///
/// Andrew W. Rose                       ///
/// 2008                                 ///
/// ////////////////////////////////////////

#include "StackedTrackerGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometryBuilder.h"


StackedTrackerGeometryESModule::StackedTrackerGeometryESModule( const edm::ParameterSet & p )
  : radial_window( p.getParameter<double>("radial_window") ),
    phi_window( p.getParameter<double>("phi_window") ),
    z_window( p.getParameter<double>("z_window") ),
    truncation_precision( p.getParameter<unsigned int>("truncation_precision") ),
    makeDebugFile( p.getParameter<bool>("make_debug_file") )
{
  setWhatProduced( this );
}

StackedTrackerGeometryESModule::~StackedTrackerGeometryESModule() {}

boost::shared_ptr <StackedTrackerGeometry> StackedTrackerGeometryESModule::produce( const StackedTrackerGeometryRecord & record )
{
  edm::ESHandle<TrackerGeometry> trkGeomHandle;
  record.getRecord<TrackerDigiGeometryRecord>().get(trkGeomHandle);

  StackedTrackerGeometryBuilder builder;
  _tracker  = boost::shared_ptr<StackedTrackerGeometry>( builder.build( &(*trkGeomHandle),
									radial_window,
									phi_window,
									z_window,
									truncation_precision,
									makeDebugFile ) );
  return _tracker;
}

DEFINE_FWK_EVENTSETUP_MODULE(StackedTrackerGeometryESModule);




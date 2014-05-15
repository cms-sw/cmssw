/*! \class   StackedTrackerGeometryESModule
 *  \brief   StackedTrackerGeometry builder
 *  \details
 *
 *  \author Andrew W. Rose
 *  \author Ivan Reid
 *  \date   2008
 *
 */

#include "StackedTrackerGeometryESModule.h"
#include "Geometry/TrackerGeometryBuilder/interface/StackedTrackerGeometryBuilder.h"

StackedTrackerGeometryESModule::StackedTrackerGeometryESModule( const edm::ParameterSet & p )
  : radial_window( p.getParameter< double >("radial_window") ),
    phi_window( p.getParameter< double >("phi_window") ),
    z_window( p.getParameter< double >("z_window") ),
    truncation_precision( p.getParameter< unsigned int >("truncation_precision") ),
    makeDebugFile( p.getParameter< bool >("make_debug_file") )
{

  /// CBC3 switch
  if ( p.exists("partitionsPerRoc") )
  {
    theNumPartitions = p.getParameter< int >("partitionsPerRoc");
    theMaxStubs = p.getParameter< unsigned int >("CBC3_MaxStubs");
    setBarrelCut = p.getParameter< std::vector< double > >("BarrelCut");

    std::vector< edm::ParameterSet > vPSet = p.getParameter< std::vector< edm::ParameterSet > >("EndcapCutSet");
    std::vector< edm::ParameterSet >::const_iterator iPSet;
    for ( iPSet = vPSet.begin(); iPSet != vPSet.end(); iPSet++ )
    {
      setRingCut.push_back( iPSet->getParameter< std::vector< double > >("EndcapCut") );
    }
  }
  else
  {
    theNumPartitions = 0;
    theMaxStubs = 0;
  }

  setWhatProduced( this );
}

StackedTrackerGeometryESModule::~StackedTrackerGeometryESModule() {}

boost::shared_ptr< StackedTrackerGeometry > StackedTrackerGeometryESModule::produce( const StackedTrackerGeometryRecord & record )
{
  edm::ESHandle< TrackerGeometry > trkGeomHandle;
  record.getRecord< TrackerDigiGeometryRecord >().get(trkGeomHandle);

  StackedTrackerGeometryBuilder builder;

  /// CBC3 switch
  if (theNumPartitions != 0)
  {
    _tracker = boost::shared_ptr< StackedTrackerGeometry >( builder.build( &(*trkGeomHandle),
                                                                           radial_window,
                                                                           phi_window,
                                                                           z_window,
                                                                           truncation_precision,
                                                                           theNumPartitions,
                                                                           theMaxStubs,
                                                                           setBarrelCut,
                                                                           setRingCut,
                                                                           makeDebugFile ) );
  }
  else
  {
    _tracker = boost::shared_ptr<StackedTrackerGeometry>( builder.build( &(*trkGeomHandle),
                                                                         radial_window,
                                                                         phi_window,
                                                                         z_window,
                                                                         truncation_precision,
                                                                         makeDebugFile ) );
  }

  return _tracker;
}

DEFINE_FWK_EVENTSETUP_MODULE(StackedTrackerGeometryESModule);


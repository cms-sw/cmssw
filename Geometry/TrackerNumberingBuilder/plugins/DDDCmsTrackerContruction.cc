#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDetIdBuilder.h"

using namespace cms;

DDDCmsTrackerContruction::DDDCmsTrackerContruction( void )
{}

const GeometricDet*
DDDCmsTrackerContruction::construct( const DDCompactView* cpv, std::vector<int> detidShifts)
{
  attribute = "TkDDDStructure"; // could come from .orcarc
  std::string value = "any";
  DDSpecificsFilter filter;
  DDValue ddv( attribute, value, 0 );
  filter.setCriteria( ddv, DDCompOp::not_equals );
  
  DDFilteredView fv( *cpv ); 
  fv.addFilter( filter );
  if( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString(attribute,&fv)) != GeometricDet::Tracker )
  {
    fv.firstChild();
    if( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString(attribute,&fv)) != GeometricDet::Tracker )
    {  
      throw cms::Exception( "Configuration" ) << " The first child of the DDFilteredView is not what is expected \n"
					      << ExtractStringFromDDD::getString( attribute, &fv ) << "\n";
    }
  }
  
  GeometricDet* tracker = new GeometricDet( &fv, GeometricDet::Tracker );
  CmsTrackerBuilder theCmsTrackerBuilder;
  theCmsTrackerBuilder.build( fv, tracker, attribute );
  
  CmsTrackerDetIdBuilder theCmsTrackerDetIdBuilder( detidShifts );
  
  tracker = theCmsTrackerDetIdBuilder.buildId( tracker );
  fv.parent();
  //
  // set the Tracker
  //
  //TrackerMapDDDtoID::instance().setTracker(tracker);
  //NOTE: If it is decided that the TrackerMapDDDtoID should be
  // constructed here, then we should return from this
  // function so that the EventSetup can manage it

  return tracker;
}


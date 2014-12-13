#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPixelPhase2EndcapBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerPhase1DiskBuilder.h"  
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerOTDiscBuilder.h"  
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

CmsTrackerPixelPhase2EndcapBuilder::CmsTrackerPixelPhase2EndcapBuilder()
{}

void
CmsTrackerPixelPhase2EndcapBuilder::buildComponent( DDFilteredView& fv, GeometricDet* g, std::string s )
{
  CmsTrackerPhase1DiskBuilder  theCmsTrackerPhase1DiskBuilder;   
  CmsTrackerOTDiscBuilder  theCmsTrackerOTDiscBuilder;   

  GeometricDet * subdet = new GeometricDet( &fv, theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )));
  std::string subdet_name = subdet->name();
  switch( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )))
  {
  case GeometricDet::PixelPhase2FullDisk:    
    theCmsTrackerPhase1DiskBuilder.build(fv,subdet,s);
    break;
  case GeometricDet::PixelPhase2ReducedDisk:    
    theCmsTrackerPhase1DiskBuilder.build(fv,subdet,s);
    break;
  case GeometricDet::OTPhase2Wheel:    
    theCmsTrackerOTDiscBuilder.build(fv,subdet,s);
    break;

  default:
    edm::LogError("CmsTrackerPixelPhase2EndcapBuilder")<<" ERROR - I was expecting a Disk... I got a "<<ExtractStringFromDDD::getString(s,&fv);
  }  
  
  g->addComponent(subdet);

}

void
CmsTrackerPixelPhase2EndcapBuilder::sortNS( DDFilteredView& fv, GeometricDet* det )
{
  GeometricDet::ConstGeometricDetContainer & comp = det->components();

  std::sort( comp.begin(), comp.end(), LessModZ());
  
  for( uint32_t i = 0; i < comp.size(); i++ )
  {
    det->component(i)->setGeographicalID(i+1); // Every subdetector: Inner pixel first, OT later, then sort by disk number
  }
}


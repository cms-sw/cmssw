#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerSubStrctBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerLayerBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerWheelBuilder.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CmsTrackerDiskBuilder.h"  
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

CmsTrackerSubStrctBuilder::CmsTrackerSubStrctBuilder()
{}

void
CmsTrackerSubStrctBuilder::buildComponent( DDFilteredView& fv, GeometricDet* g, std::string s )
{
  CmsTrackerLayerBuilder theCmsTrackerLayerBuilder;
  CmsTrackerWheelBuilder theCmsTrackerWheelBuilder;
  CmsTrackerDiskBuilder  theCmsTrackerDiskBuilder;   

  GeometricDet * subdet = new GeometricDet( &fv, theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )));
  switch( theCmsTrackerStringToEnum.type( ExtractStringFromDDD::getString( s, &fv )))
  {
  case GeometricDet::layer:
    theCmsTrackerLayerBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::OTPhase2Layer:
    theCmsTrackerLayerBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::wheel:
    theCmsTrackerWheelBuilder.build(fv,subdet,s);      
    break;
  case GeometricDet::disk:    
    theCmsTrackerDiskBuilder.build(fv,subdet,s);
    break;

  default:
    edm::LogError("CmsTrackerSubStrctBuilder")<<" ERROR - I was expecting a Layer ,Wheel or Disk... I got a "<<ExtractStringFromDDD::getString(s,&fv);
  }  
  
  g->addComponent(subdet);

}

void
CmsTrackerSubStrctBuilder::sortNS( DDFilteredView& fv, GeometricDet* det )
{
  GeometricDet::ConstGeometricDetContainer & comp = det->components();

  switch( comp.front()->type())
  {
  case GeometricDet::layer:
    std::sort( comp.begin(), comp.end(), LessR());
    break;	
  case GeometricDet::OTPhase2Layer:
    std::sort( comp.begin(), comp.end(), LessR());
    break;  
  case GeometricDet::wheel:
    std::sort( comp.begin(), comp.end(), LessModZ());
    break;	
  case GeometricDet::disk:
    std::sort( comp.begin(), comp.end(), LessModZ());
    break;
  default:
    edm::LogError( "CmsTrackerSubStrctBuilder" ) << "ERROR - wrong SubDet to sort..... " << det->components().front()->type(); 
  }
  
  for( uint32_t i = 0; i < comp.size(); i++ )
  {
    det->component(i)->setGeographicalID(i+1); // Every subdetector: Layer/Disk/Wheel Number
  }
}


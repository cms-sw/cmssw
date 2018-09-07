#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDEndcapBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDDiscBuilder.h"  
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

CmsMTDEndcapBuilder::CmsMTDEndcapBuilder()
{}

void
CmsMTDEndcapBuilder::buildComponent( DDFilteredView& fv, GeometricTimingDet* g, std::string s )
{
  CmsMTDDiscBuilder  theCmsMTDDiscBuilder;   
  
  GeometricTimingDet * subdet = new GeometricTimingDet( &fv, theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ));
  std::string subdet_name = subdet->name().fullname();
  switch( theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ))
  {  
  case GeometricTimingDet::ETLDisc:    
    theCmsMTDDiscBuilder.build(fv,subdet,s);
    break;
  default:
    throw cms::Exception("CmsMTDEndcapBuilder")<<" ERROR - I was expecting a Disk... I got a "<< fv.logicalPart().name().fullname();
  }  

  g->addComponent(subdet);
}


#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
void
CmsMTDEndcapBuilder::sortNS( DDFilteredView& fv, GeometricTimingDet* det )
{
  GeometricTimingDet::ConstGeometricTimingDetContainer & comp = det->components();

  std::stable_sort( comp.begin(), comp.end(), LessModZ());
  
  for( uint32_t i = 0; i < comp.size(); i++ )
  {
    const uint32_t side = det->component(i)->translation().z() > 0 ? 1 : 0;
    det->component(i)->setGeographicalID(ETLDetId(side,0,0,0)); 
  }
}


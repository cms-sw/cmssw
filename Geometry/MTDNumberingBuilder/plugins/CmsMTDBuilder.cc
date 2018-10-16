#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDSubStrctBuilder.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDEndcapBuilder.h"

#include <bitset>

CmsMTDBuilder::CmsMTDBuilder()
{}

void
CmsMTDBuilder::buildComponent( DDFilteredView& fv, GeometricTimingDet* g, std::string s )
{
  CmsMTDSubStrctBuilder theCmsMTDSubStrctBuilder;
  CmsMTDEndcapBuilder theCmsMTDEndcapBuilder;
  
  GeometricTimingDet* subdet = new GeometricTimingDet( &fv, theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ) );
  
  switch( theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ) )
  {  
  case GeometricTimingDet::ETL:
    theCmsMTDEndcapBuilder.build( fv, subdet, s );      
    break;  
  case GeometricTimingDet::BTL:
    theCmsMTDSubStrctBuilder.build( fv, subdet, s ); 
    break;  
  default:
    throw cms::Exception("CmsMTDBuilder") << " ERROR - I was expecting a SubDet, I got a " << fv.logicalPart().name().fullname();   
  }
  
  g->addComponent( subdet );
}

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
void
CmsMTDBuilder::sortNS( DDFilteredView& fv, GeometricTimingDet* det )
{  
  GeometricTimingDet::ConstGeometricTimingDetContainer & comp = det->components();
  std::stable_sort( comp.begin(), comp.end(), subDetByType);
  
  for( uint32_t i = 0; i < comp.size(); i++ )
  {
    const uint32_t side = det->component(i)->translation().z() > 0 ? 1 : 0;
    switch( comp[i]->type() ) { 
    case GeometricTimingDet::BTL:
      det->component(i)->setGeographicalID(BTLDetId(0,0,0,0,0));  
      break;
    case GeometricTimingDet::ETL:      
      det->component(i)->setGeographicalID(ETLDetId(side,0,0,0));  
      break;
    default:
      throw cms::Exception("CmsMTDBuilder") << " ERROR - I was expecting a SubDet, I got a " << comp[i]->name(); 
    }
  }
}





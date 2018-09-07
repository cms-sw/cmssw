#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDSubStrctBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDTrayBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>

#include <bitset>

namespace {
  constexpr std::array<const char*,2> sides{ { "PositiveZ","NegativeZ" } };
}

CmsMTDSubStrctBuilder::CmsMTDSubStrctBuilder()
{}

void
CmsMTDSubStrctBuilder::buildComponent( DDFilteredView& fv, GeometricTimingDet* g, std::string side )
{
  CmsMTDTrayBuilder theCmsMTDTrayBuilder;

  GeometricTimingDet * subdet = new GeometricTimingDet( &fv, theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ));
  
  switch( theCmsMTDStringToEnum.type( fv.logicalPart().name().fullname() ) ) {  
  case GeometricTimingDet::BTLLayer:
    theCmsMTDTrayBuilder.build(fv,subdet,side);      
    break;  
    
  default:
    throw cms::Exception("CmsMTDSubStrctBuilder")<<" ERROR - I was expecting a BTLLayer... I got a "<< fv.logicalPart().name().fullname();
  }  
  
  g->addComponent(subdet);
}

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
void
CmsMTDSubStrctBuilder::sortNS( DDFilteredView& fv, GeometricTimingDet* det )
{
  GeometricTimingDet::ConstGeometricTimingDetContainer & comp = det->components();

  switch( comp.front()->type())
  {  
  case GeometricTimingDet::BTLLayer:
    std::stable_sort( comp.begin(), comp.end(), LessR());
    break;    
  default:
    edm::LogError( "CmsMTDSubStrctBuilder" ) << "ERROR - wrong SubDet to sort..... " << det->components().front()->type(); 
  }
  
  for( uint32_t i = 0; i < comp.size(); i++ )
  {
    det->component(i)->setGeographicalID(i+1); // Every subdetector: Layer/Disk/Wheel Number
  }
}


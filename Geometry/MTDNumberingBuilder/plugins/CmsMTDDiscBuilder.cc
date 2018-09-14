#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDDiscBuilder.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "Geometry/MTDNumberingBuilder/plugins/ExtractStringFromDDD.h"
#include "Geometry/MTDNumberingBuilder/plugins/CmsMTDETLRingBuilder.h"
#include "Geometry/MTDNumberingBuilder/plugins/MTDStablePhiSort.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <vector>
#include <algorithm>

using namespace std;

void
CmsMTDDiscBuilder::buildComponent( DDFilteredView& fv, GeometricTimingDet* g, std::string s )
{
  CmsMTDETLRingBuilder theCmsMTDETLRingBuilder;
  const std::string ringname = fv.logicalPart().name().fullname().substr(0,8);
  GeometricTimingDet * subdet = new GeometricTimingDet( &fv, theCmsMTDStringToEnum.type( ringname ));

  switch( theCmsMTDStringToEnum.type( ringname ))
  {
  case GeometricTimingDet::ETLRing:
    theCmsMTDETLRingBuilder.build( fv, subdet, s );    
    break;
  default:
    throw cms::Exception( "CmsMTDDiscBuilder" ) << " ERROR - I was expecting a Ring, I got a " <<  ringname;   
  }  
  
  g->addComponent( subdet );
}

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
void
CmsMTDDiscBuilder::sortNS( DDFilteredView& fv, GeometricTimingDet* det )
{


  GeometricTimingDet::ConstGeometricTimingDetContainer & comp = det->components();

  switch(det->components().front()->type()){
  case GeometricTimingDet::ETLRing:
    std::stable_sort(comp.begin(),comp.end(),isLessRModule);
    break;
  default:
    edm::LogError("CmsMTDDiscBuilder")<<"ERROR - wrong SubDet to sort..... "<<det->components().front()->type();
  }
  
  GeometricTimingDet::GeometricTimingDetContainer rings;
  uint32_t totalrings = comp.size();

  const uint32_t side = det->translation().z() > 0 ? 1 : 0;

  for ( uint32_t rn=0; rn<totalrings; ++rn) {    
    det->component(rn)->setGeographicalID(ETLDetId(side,rn+1,0,0));
  }

}


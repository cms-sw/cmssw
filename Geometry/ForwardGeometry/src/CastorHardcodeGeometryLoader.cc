#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorHardcodeGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "CastorGeometryData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

CastorHardcodeGeometryLoader::CastorHardcodeGeometryLoader() {
  init();
}

CastorHardcodeGeometryLoader::CastorHardcodeGeometryLoader(const CastorTopology& ht) : 
  theTopology(ht) {
  init();
}

void CastorHardcodeGeometryLoader::init() {
  theEMSectiondX = 2.*dXEMPlate;
  theEMSectiondY = 2.*dYEMPlate;
  theEMSectiondZ = 101.0;
  theHADSectiondX = 2.*dXHADPlate; 
  theHADSectiondY = 2.*dYHADPlate;
  theHADSectiondZ = 1212.;
}

std::auto_ptr<CaloSubdetectorGeometry> CastorHardcodeGeometryLoader::load(DetId::Detector det, int subdet){
  std::auto_ptr<CaloSubdetectorGeometry> hg(new CastorGeometry(&theTopology));
  if(subdet == HcalCastorDetId::SubdetectorId){
    fill(HcalCastorDetId::EM,hg.get());
    fill(HcalCastorDetId::HAD,hg.get());
  }
  return hg;
}

std::auto_ptr<CaloSubdetectorGeometry> CastorHardcodeGeometryLoader::load() {
  std::auto_ptr<CaloSubdetectorGeometry> hg(new CastorGeometry(&theTopology));
  fill(HcalCastorDetId::EM,hg.get());
  fill(HcalCastorDetId::HAD,hg.get());
  return hg;
}

void CastorHardcodeGeometryLoader::fill(HcalCastorDetId::Section section, CaloSubdetectorGeometry* geom) 
{
  // start by making the new HcalDetIds
  std::vector<HcalCastorDetId> castorIds;
  HcalCastorDetId id;
  int firstCell = theTopology.firstCell(section);
  int lastCell = theTopology.lastCell(section);
  for(int imodule = firstCell; imodule <= lastCell; ++imodule) {
    for(int isector = 1; isector < 17; ++isector) {
    id = HcalCastorDetId(section, true, isector, imodule);
    if(theTopology.valid(id)) castorIds.push_back(id);
    id = HcalCastorDetId(section, false, isector, imodule);
    if(theTopology.valid(id)) castorIds.push_back(id);
   }
  }
  edm::LogInfo("CastorHardcodeGeometry") << "Number of Castor DetIds made: " << section << " " << castorIds.size();
 
  // for each new HcalCastorDetId, make a CaloCellGeometry

 for(std::vector<HcalCastorDetId>::const_iterator castorIdItr = castorIds.begin();
     castorIdItr != castorIds.end(); ++castorIdItr)
   {
     const CaloCellGeometry * geometry = makeCell(*castorIdItr, geom );
     geom->addCell(*castorIdItr, geometry);
   }
}

const CaloCellGeometry*
CastorHardcodeGeometryLoader::makeCell(const HcalCastorDetId & detId, CaloSubdetectorGeometry* geom) const {
  
  float zside = detId.zside();
  HcalCastorDetId::Section section = detId.section();
  int module = detId.module();
//  int sector = detId.sector();
  
  float xMother = X0;
  float yMother = Y0;
  float zMother = Z0;
  float dx = 0;
  float dy = 0;
  float dz = 0;
  float theTiltAngle = 0.7854;
  float xfaceCenter = 0;
  float yfaceCenter = 0;
  float zfaceCenter = 0;

  if(section==HcalCastorDetId::EM){
    dx = theEMSectiondX;
    dy = theEMSectiondY;
    dz = theEMSectiondZ;
//    theTiltAngle = 0.0;
    xfaceCenter = xMother + (theXChannelBoundaries[module-1] + theEMSectiondX/2);
    yfaceCenter = yMother; 
    zfaceCenter = (zMother + theZSectionBoundaries[0])*zside;
   }

  if(section==HcalCastorDetId::HAD){
    dx = theHADSectiondX;
    dy = theHADSectiondY;
    dz = theHADSectiondZ;
    theTiltAngle = tiltangle;
    xfaceCenter = xMother;
    yfaceCenter = yMother; 
    zfaceCenter = (zMother + theHadmodulesBoundaries[module-1])*zside;
  }
  GlobalPoint faceCenter(xfaceCenter, yfaceCenter, zfaceCenter);

  std::vector<float> zz ;
  zz.resize(3) ;
  zz.push_back( dx ) ;
  zz.push_back( dy ) ;
  zz.push_back( dz ) ;
  return new calogeom::IdealCastorTrapezoid( 
     faceCenter, 
     geom->cornersMgr(),
     CaloCellGeometry::getParmPtr( zz, 
				   geom->parMgr(), 
				   geom->parVecVec() ) );
//     CaloCellGeometry::getParmPtr( zz, 3, geom->parVecVec() ) );
}



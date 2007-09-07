#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcHardcodeGeometryLoader.h"
#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "ZdcHardcodeGeometryData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

ZdcHardcodeGeometryLoader::ZdcHardcodeGeometryLoader() {
  init();
}

ZdcHardcodeGeometryLoader::ZdcHardcodeGeometryLoader(const ZdcTopology& ht) : 
  theTopology(ht) {
  init();
}

void ZdcHardcodeGeometryLoader::init() {
  theEMSectiondX = 2.*dXPlate;
  theEMSectiondY = 2.*dYPlate;
  theEMSectiondZ = 99.0;
  theLUMSectiondX = 2.*dXPlate; 
  theLUMSectiondY = 2*dYLUM;
  theLUMSectiondZ = 94.0;
  theHADSectiondX = 2.*dXPlate; 
  theHADSectiondY = 2.*dYPlate;
  theHADSectiondZ = 139.2;
}

std::auto_ptr<CaloSubdetectorGeometry> ZdcHardcodeGeometryLoader::load(DetId::Detector det, int subdet){
  std::auto_ptr<CaloSubdetectorGeometry> hg(new ZdcGeometry(&theTopology));
  if(subdet == HcalZDCDetId::SubdetectorId){
    fill(HcalZDCDetId::EM,hg.get());
    fill(HcalZDCDetId::LUM,hg.get());
    fill(HcalZDCDetId::HAD,hg.get());
  }
  return hg;
}

std::auto_ptr<CaloSubdetectorGeometry> ZdcHardcodeGeometryLoader::load() {
  std::auto_ptr<CaloSubdetectorGeometry> hg(new ZdcGeometry(&theTopology));
  fill(HcalZDCDetId::EM,hg.get());
  fill(HcalZDCDetId::LUM,hg.get());
  fill(HcalZDCDetId::HAD,hg.get());
  return hg;
}

void ZdcHardcodeGeometryLoader::fill(HcalZDCDetId::Section section, CaloSubdetectorGeometry* geom) 
{
  // start by making the new HcalDetIds
  std::vector<HcalZDCDetId> zdcIds;
  HcalZDCDetId id;
  int firstCell = theTopology.firstCell(section);
  int lastCell = theTopology.lastCell(section);
  for(int idepth = firstCell; idepth <= lastCell; ++idepth) {
    id = HcalZDCDetId(section, true, idepth);
    if(theTopology.valid(id)) zdcIds.push_back(id);
    id = HcalZDCDetId(section, false, idepth);
    if(theTopology.valid(id)) zdcIds.push_back(id);
   }
  edm::LogInfo("ZdcHardcodeGeometry") << "Number of ZDC DetIds made: " << section << " " << zdcIds.size();
 
  // for each new HcalZdcDetId, make a CaloCellGeometry

 for(std::vector<HcalZDCDetId>::const_iterator zdcIdItr = zdcIds.begin();
     zdcIdItr != zdcIds.end(); ++zdcIdItr)
   {
     const CaloCellGeometry * geometry = makeCell(*zdcIdItr, geom );
     geom->addCell(*zdcIdItr, geometry);
   }
}

const CaloCellGeometry*
ZdcHardcodeGeometryLoader::makeCell(const HcalZDCDetId & detId,
				    CaloSubdetectorGeometry* geom) const {
  
  float zside = detId.zside();
  HcalZDCDetId::Section section = detId.section();
  int channel = detId.depth();
  
  float xMother = X0;
  float yMother = Y0;
  float zMother = Z0;
  float dx = 0;
  float dy = 0;
  float dz = 0;
  float theTiltAngle = 0;
  float xfaceCenter = 0;
  float yfaceCenter = 0;
  float zfaceCenter = 0;

  if(section==HcalZDCDetId::EM){
    dx = theEMSectiondX;
    dy = theEMSectiondY;
    dz = theEMSectiondZ;
    theTiltAngle = 0.0;
    xfaceCenter = xMother + (theXChannelBoundaries[channel-1] + theEMSectiondX/2);
    yfaceCenter = yMother; 
    zfaceCenter = (zMother + theZSectionBoundaries[0])*zside;
  }

  if(section==HcalZDCDetId::LUM){
    dx = theLUMSectiondX;
    dy = theLUMSectiondY;
    dz = theLUMSectiondZ;
    theTiltAngle = 0.0;
    xfaceCenter = xMother;
    yfaceCenter = yMother + YLUM; 
    zfaceCenter = (zMother + theZLUMChannelBoundaries[channel-1])*zside;
  }

  if(section==HcalZDCDetId::HAD){
    dx = theHADSectiondX;
    dy = theHADSectiondY;
    dz = theHADSectiondZ;
    theTiltAngle = tiltangle;
    xfaceCenter = xMother;
    yfaceCenter = yMother; 
    zfaceCenter = (zMother + theZLUMChannelBoundaries[channel-1])*zside;
  }
  GlobalPoint faceCenter(xfaceCenter, yfaceCenter, zfaceCenter);

  std::vector<double> zz ;
  zz.resize(3) ;
  zz.push_back( dx ) ;
  zz.push_back( dy ) ;
  zz.push_back( dz ) ;
  return new calogeom::IdealZDCTrapezoid( 
     faceCenter, 
     geom->cornersMgr(),
     CaloCellGeometry::getParmPtr( zz, 3, geom->parVecVec() ) );
}



#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "CastorGeometryData.h"

CastorGeometry::CastorGeometry(const CastorTopology * topology)
: theTopology(topology), 
  lastReqDet_(DetId::Detector(0)),
  lastReqSubdet_(0)
{
}


CastorGeometry::~CastorGeometry() {
}

std::vector<DetId> const & CastorGeometry::getValidDetIds(DetId::Detector det, int subdet) {
  if (lastReqDet_!=det || lastReqSubdet_!=subdet) {
    lastReqDet_=det;
    lastReqSubdet_=subdet;
    m_validIds.clear();
  }
  if (m_validIds.empty()) {
    m_validIds.reserve(cellGeometries().size());
    CaloSubdetectorGeometry::CellCont::const_iterator i;
    for (i=cellGeometries().begin(); i!=cellGeometries().end(); i++) {
      DetId id(i->first);
      if (id.det()==det && id.subdetId()==subdet)
        m_validIds.push_back(id);
    }
      std::sort(m_validIds.begin(),m_validIds.end());
  }

  return m_validIds;
}

/*  NOTE only draft implementation at the moment
    what about dead volumes?
*/

DetId CastorGeometry::getClosestCell(const GlobalPoint& r) const
{
  // first find the side
  double z = r.z();
//  double x = r.x();
  double y = r.y();
  double dz = 0.;
  double zt = 0.;
  double phi = r.phi();

  int zside = 0;
  if(z >= 0)
    zside = 1;
  else
    zside =-1;

  bool isPositive = false;
  if(z>0)isPositive = true;
  z = fabs(z);
  
  // figure out if it's EM or HAD section
  // EM length = 2x51.5 mm,  HAD length = 12x101 mm
  // I assume that z0 of Castor is 14385 mm (cms.xml)
  HcalCastorDetId::Section section = HcalCastorDetId::EM;
  if(z<= theZSectionBoundaries[1])section = HcalCastorDetId::EM;
  if(z>theZSectionBoundaries[2])section = HcalCastorDetId::HAD;

  ///////////
  // figure out sector: 1-16
  // in CastorGeometryData.h theSectorBoundaries define the phi range of sectors
  //////////////
  int sector = -1;
  if(theSectorBoundaries[1]<= phi <theSectorBoundaries[2]) sector =1;
  if(theSectorBoundaries[2]<= phi <theSectorBoundaries[3]) sector =2;
  if(theSectorBoundaries[3]<= phi <theSectorBoundaries[4]) sector =3;
  if(theSectorBoundaries[4]<= phi <theSectorBoundaries[5]) sector =4;
  if(theSectorBoundaries[5]<= phi <theSectorBoundaries[6]) sector =5;
  if(theSectorBoundaries[6]<= phi <theSectorBoundaries[7]) sector =6;
  if(theSectorBoundaries[7]<= phi <theSectorBoundaries[8]) sector =7;
  if(theSectorBoundaries[8]<= phi <theSectorBoundaries[9]) sector =8;
  if(theSectorBoundaries[9]<= phi <theSectorBoundaries[10]) sector =9;
  if(theSectorBoundaries[10]<= phi <theSectorBoundaries[11]) sector =10;
  if(theSectorBoundaries[11]<= phi <theSectorBoundaries[12]) sector =11;
  if(theSectorBoundaries[12]<= phi <theSectorBoundaries[13]) sector =12;
  if(theSectorBoundaries[13]<= phi <theSectorBoundaries[14]) sector =13;
  if(theSectorBoundaries[14]<= phi <theSectorBoundaries[15]) sector =14;
  if(theSectorBoundaries[15]<= phi <theSectorBoundaries[16]) sector =15;
  if(theSectorBoundaries[16]<= phi) sector =16;

//figure out module, just a draft for checks
  int module = -1;

// NOTE check 
 if(section ==HcalCastorDetId::EM){
  if(fabs(y) > dYEMPlate*sin(tiltangle))
      dz = (y > 0.) ?  dYEMPlate*cos(tiltangle) : -  dYEMPlate*sin(tiltangle);
    else
      dz = (y > 0.) ?  y/tan(tiltangle) : -y/tan(tiltangle);
    zt = z - dz;
    if(theZSectionBoundaries[1]<= zt <theZSectionBoundaries[2]) module = 1;
    if(zt > (theZSectionBoundaries[1]-51.5) )module = 2;
  }

  if(section == HcalCastorDetId::HAD){
    if(fabs(y) > dYHADPlate*sin(tiltangle))
      dz = (y > 0.) ?  dYHADPlate*cos(tiltangle) : -  dYHADPlate*sin(tiltangle);
    else
      dz = (y > 0.) ?  y/tan(tiltangle) : -y/tan(tiltangle);
    zt = z - dz;
    if(zt< theHadmodulesBoundaries[1]) module = 1;
    if(theHadmodulesBoundaries[1]<= zt <theHadmodulesBoundaries[2]) module = 2;
    if(theHadmodulesBoundaries[2]<= zt <theHadmodulesBoundaries[3]) module = 3;
    if(theHadmodulesBoundaries[4]<= zt <theHadmodulesBoundaries[5]) module = 4;
    if(theHadmodulesBoundaries[5]<= zt <theHadmodulesBoundaries[6]) module = 5;
    if(theHadmodulesBoundaries[6]<= zt <theHadmodulesBoundaries[7]) module = 6;
    if(theHadmodulesBoundaries[7]<= zt <theHadmodulesBoundaries[8]) module = 7;
    if(theHadmodulesBoundaries[8]<= zt <theHadmodulesBoundaries[9]) module = 8;
    if(theHadmodulesBoundaries[9]<= zt <theHadmodulesBoundaries[10]) module = 9;
    if(theHadmodulesBoundaries[10]<= zt <theHadmodulesBoundaries[11]) module = 10;
    if(theHadmodulesBoundaries[11]<= zt <theHadmodulesBoundaries[12]) module = 11;
    if(theHadmodulesBoundaries[12]<= zt ) module = 12;
  }
  
  HcalCastorDetId bestId  = HcalCastorDetId(section,isPositive, sector, module);
  return bestId;
}



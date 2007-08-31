#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "ZdcHardcodeGeometryData.h"

ZdcGeometry::ZdcGeometry(const ZdcTopology * topology) 
: theTopology(topology),
  lastReqDet_(DetId::Detector(0)), 
  lastReqSubdet_(0) 
{
}

ZdcGeometry::~ZdcGeometry() {
}

std::vector<DetId> const & ZdcGeometry::getValidDetIds(DetId::Detector det, int subdet) const {
  if (lastReqDet_!=det || lastReqSubdet_!=subdet) {
    lastReqDet_=det;
    lastReqSubdet_=subdet;
    validIds_.clear();
  }
  if (validIds_.empty()) {
    validIds_.reserve(cellGeometries().size());
    CaloSubdetectorGeometry::CellCont::const_iterator i;
    for (i=cellGeometries().begin(); i!=cellGeometries().end(); i++) {
      DetId id(i->first);
      if (id.det()==det && id.subdetId()==subdet) 
	validIds_.push_back(id);
    }
      std::sort(validIds_.begin(),validIds_.end());
  }

  return validIds_;
}


DetId ZdcGeometry::getClosestCell(const GlobalPoint& r) const
{
  // first find the side
  double z = r.z();
  double x = r.x();
  double y = r.y();
  double dz = 0.;
  double zt = 0.;

  int zside = 0;
  if(z >= 0)
    zside = 1;
  else
    zside =-1;

  bool isPositive = false;
  if(z>0)isPositive = true;
  z = fabs(z);
  
  // figure out if is closer to EM, HAD or LUM section
  HcalZDCDetId::Section section = HcalZDCDetId::Unknown;
  if(z<= theZSectionBoundaries[1])section = HcalZDCDetId::EM;
  if(theZSectionBoundaries[1]<z<= theZSectionBoundaries[2])section = HcalZDCDetId::LUM;
  if(z>theZSectionBoundaries[2])section = HcalZDCDetId::HAD;

  // figure out channel
  int channel = -1;
  if(section ==HcalZDCDetId::EM){
    if(x < theXChannelBoundaries[1]) channel = 1;
    if(theXChannelBoundaries[1]<= x <theXChannelBoundaries[2])channel = 2;
    if(theXChannelBoundaries[2]<= x <theXChannelBoundaries[3])channel = 3;
    if(theXChannelBoundaries[3]<= x <theXChannelBoundaries[4])channel = 4;
    if(x > theXChannelBoundaries[4])channel = 5;
  }
  
  if(section == HcalZDCDetId::LUM){
    if(z <= theZLUMChannelBoundaries[1])channel = 1;
    if(z > theZLUMChannelBoundaries[1])channel = 2;
  } 
  if(section == HcalZDCDetId::HAD){
    if(fabs(y) > dYPlate*sin(tiltangle))
      dz = (y > 0.) ?  dYPlate*cos(tiltangle) : -  dYPlate*sin(tiltangle);
    else
      dz = (y > 0.) ?  y/tan(tiltangle) : -y/tan(tiltangle); 
    zt = z - dz;
    if(zt< theZHadChannelBoundaries[1]) channel = 1;
    if(theZHadChannelBoundaries[1]<=  zt <theZHadChannelBoundaries[2])channel = 2;
    if(theZHadChannelBoundaries[2]<=  zt <theZHadChannelBoundaries[3])channel = 3;
    if(zt > theZHadChannelBoundaries[4])channel = 4;
  }
  
  HcalZDCDetId bestId  = HcalZDCDetId(section,isPositive,channel);
  return bestId;
}


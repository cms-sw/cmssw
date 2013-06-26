//FAMOS headers
#include "FastSimulation/CaloGeometryTools/interface/CaloPoint.h"

std::ostream & operator<<(std::ostream& ost ,const CaloPoint& cid)
{
  ost << " DetId " ;
  if(!cid.getDetId().null())
    ost <<  (uint32_t)cid.getDetId()();
  else
    ost << cid.whichDetector();
  //  ost << " Side " << cid.getSide() << " Point " << (XYZPoint)cid;
  ost << " Point " << (math::XYZVector)cid;
  return ost;
}

// For the ECAL
CaloPoint::CaloPoint(const DetId & cell, CaloDirection side, const XYZPoint& position):
  XYZPoint(position),cellid_(cell),side_(side)
{
  detector_=cell.det();
  subdetector_=cell.subdetId();
  layer_=0;
}

//hcal
CaloPoint::CaloPoint(DetId::Detector det,const XYZPoint& position)
  :XYZPoint(position),detector_(det)
{
  subdetector_=0;
  layer_=0;
}

//preshower
CaloPoint::CaloPoint(DetId::Detector detector,int subdetn, int layer,const XYZPoint& position)
  :XYZPoint(position),detector_(detector),subdetector_(subdetn),layer_(layer)
{
  cellid_=DetId();
}

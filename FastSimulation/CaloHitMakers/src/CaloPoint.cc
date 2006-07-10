//FAMOS headers
#include "FastSimulation/CaloHitMakers/interface/CaloPoint.h"

std::ostream & operator<<(std::ostream& ost ,const CaloPoint& cid)
{
  ost << " DetId " ;
  if(!cid.getDetId().null())
    ost <<  (uint32_t)cid.getDetId()();
  else
    ost << cid.whichDetector();
  //  ost << " Side " << cid.getSide() << " Point " << (HepPoint3D)cid;
  ost << " Point " << (HepPoint3D)cid;
  return ost;
};


//CaloPoint::CaloPoint(CellID cell, CaloDirection side, const HepPoint3D& position)
//  :HepPoint3D(position),cellid_(cell),side_(side)
//{
//  if(cell!=CellID()) detector_=cell.whichDetector();
//}
//
//CaloPoint::CaloPoint(string detector,CaloDirection side, const HepPoint3D& position)
//  :HepPoint3D(position),side_(side),detector_(detector)
//{
//  cellid_=CellID();
//}

CaloPoint::CaloPoint(const Calorimeter* calo, std::string detector, const HepPoint3D& position):HepPoint3D(position),myCalorimeter_(calo),detector_(detector)
{
  cellid_=DetId();
}

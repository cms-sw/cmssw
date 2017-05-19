/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPCommon.h"
#include <iostream>
#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

char TotemRPGeometry::Build(const DetGeomDesc *gD)
{
  // propagate through the GeometricalDet structure and add
  // all detectors to 'theMap'
  deque<const DetGeomDesc *> buffer;
  buffer.push_back(gD);
  while (buffer.size() > 0)
  {
    const DetGeomDesc *d = buffer.front();
    buffer.pop_front();

    // check if it is RP detector
    if (! d->name().name().compare(DDD_TOTEM_RP_DETECTOR_NAME)
       or d->name().name().compare(DDD_CTPPS_DIAMONDS_DETECTOR_NAME)==0)
      AddDetector(d->geographicalID(), d);

    // check if it is RP device (primary vacuum)
    if (! d->name().name().compare(DDD_TOTEM_RP_PRIMARY_VACUUM_NAME))
      AddRPDevice(d->geographicalID(), d);
    
    for (unsigned int i = 0; i < d->components().size(); i++)
      buffer.push_back(d->components()[i]);
  }

  // build sets from theMap
  BuildSets();

  return 0;
}

//----------------------------------------------------------------------------------------------------

char TotemRPGeometry::AddDetector(unsigned int id, const DetGeomDesc* &gD)
{
  // check if the ID is already in map
  if (theMap.find(id) != theMap.end())
    return 1;

  // add gD
  theMap[id] = (DetGeomDesc*) gD;
  return 0;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc* TotemRPGeometry::GetDetector(unsigned int id) const
{
  // check if id is RP id?

  // check if there is a corresponding key
//  std::cout<<"TotemRPGeometry::GetDetector entered, id="<<id<<std::endl;
  mapType::const_iterator it = theMap.find(id);
  if (it == theMap.end())
    throw cms::Exception("TotemRPGeometry") << "Not found detector with ID " << id << ", i.e. "
      << CTPPSDetId(id);

  // the [] operator cannot be used as this method is const
  // and it must be const and one gets TotemRPGeometry const
  // from EventSetup
  //std::cout<<"det. retrieved:"<<id<<std::endl;
  return (*it).second;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GetDetEdgePosition(unsigned int id) const
{
	// hardcoded for now, values taken from RP_Hybrid.xml
	// +-------+
	// |       |
	// |   + (0,0)
	//  *(x,y) |
	//   \-----+
	// x=-RP_Det_Size_a/2+RP_Det_Edge_Length/(2*sqrt(2))
	// y=x
	// ideally we would get this from the geometry in the event setup
	double x=-36.07/2+22.276/(2*sqrt(2));
	return LocalToGlobal(id, CLHEP::Hep3Vector(x, x, 0.));
}

// Left edge: -18.0325, -2.2209 ; Right edge: -2.2209, -18.0325

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GetDetEdgeNormalVector(unsigned int id) const
{
	return GetDetector(id)->rotation() * CLHEP::Hep3Vector(-sqrt(2)/2, -sqrt(2)/2, 0.);
}

//----------------------------------------------------------------------------------------------------

char TotemRPGeometry::AddRPDevice(unsigned int id, const DetGeomDesc* &gD)
{
  // check if the copy_no is already in map
  if (theRomanPotMap.find(id) != theRomanPotMap.end())
    return 1;

  // add gD
  theRomanPotMap[id] = (DetGeomDesc*) gD;
  return 0;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc* TotemRPGeometry::GetRPDevice(unsigned int id) const
{
  // check if there is a corresponding key
  RPDeviceMapType::const_iterator it = theRomanPotMap.find(id);
  if (it == theRomanPotMap.end())
    throw cms::Exception("TotemRPGeometry") << "Not found RP device with ID " << id << ", i.e. "
      << TotemRPDetId(id);

  return (*it).second;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GetRPThinFoilPosition(int copy_no) const
{
	// hardcoded for now, taken from RP_Box.xml:RP_Box_primary_vacuum_y
	// ideally we would get this from the geometry in the event setup
	return LocalToGlobal(GetRPDevice(copy_no), CLHEP::Hep3Vector(0., -135.65/2.0, 0.));
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GetRPThinFoilNormalVector(int copy_no) const
{
	return GetRPDevice(copy_no)->rotation() * CLHEP::Hep3Vector(0., -1., 0.);
}

//----------------------------------------------------------------------------------------------------

void TotemRPGeometry::BuildSets()
{
  // reset
  stationsInArm.clear();
  rpsInStation.clear();
  detsInRP.clear();

  // build
  for (mapType::const_iterator it = theMap.begin(); it != theMap.end(); ++it)
  {
    const CTPPSDetId detId(it->first);
    const CTPPSDetId rpId = detId.getRPId();
    const CTPPSDetId stId = detId.getStationId();
    const CTPPSDetId armId = detId.getArmId();

    stationsInArm[armId].insert(armId);
    rpsInStation[stId].insert(rpId);
    detsInRP[rpId].insert(detId);
  }
}

//----------------------------------------------------------------------------------------------------
std::set<unsigned int> TotemRPGeometry::StationsInArm(unsigned int id) const
{
  mapSetType::const_iterator it = stationsInArm.find(id);
  if (it == stationsInArm.end())
    throw cms::Exception("TotemRPGeometry") << "Arm with ID " << id << " not found.";
  return (*it).second;
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int> TotemRPGeometry::RPsInStation(unsigned int id) const
{
  mapSetType::const_iterator it = rpsInStation.find(id);
  if (it == rpsInStation.end())
    throw cms::Exception("TotemRPGeometry") << "Station with ID " << id << " not found.";
  return (*it).second;
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int> TotemRPGeometry::DetsInRP(unsigned int id) const
{
  mapSetType::const_iterator it = detsInRP.find(id);
  if (it == detsInRP.end())
    throw cms::Exception("TotemRPGeometry") << "RP with ID " << id << " not found.";
  return (*it).second;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::LocalToGlobal(DetGeomDesc *gd, const CLHEP::Hep3Vector r) const
{
  CLHEP::Hep3Vector tmp = gd->rotation() * r;
  tmp.setX(tmp.x() + (gd->translation()).x());
  tmp.setY(tmp.y() + (gd->translation()).y());
  tmp.setZ(tmp.z() + (gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::LocalToGlobal(unsigned int id, const CLHEP::Hep3Vector r) const
{
  DetGeomDesc *gd = GetDetector(id);
  CLHEP::Hep3Vector tmp = gd->rotation() * r;
  tmp.setX(tmp.x() + (gd->translation()).x());
  tmp.setY(tmp.y() + (gd->translation()).y());
  tmp.setZ(tmp.z() + (gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GlobalToLocal(DetGeomDesc *gd, const CLHEP::Hep3Vector r) const
{
  CLHEP::Hep3Vector tmp = r;
  tmp.setX(tmp.x() - (gd->translation()).x());
  tmp.setY(tmp.y() - (gd->translation()).y());
  tmp.setZ(tmp.z() - (gd->translation()).z());
  return (gd->rotation()).Inverse() * tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GlobalToLocal(unsigned int id, const CLHEP::Hep3Vector r) const
{
  DetGeomDesc *gd = GetDetector(id);
  CLHEP::Hep3Vector tmp = r;
  tmp.setX(tmp.x() - (gd->translation()).x());
  tmp.setY(tmp.y() - (gd->translation()).y());
  tmp.setZ(tmp.z() - (gd->translation()).z());
  return (gd->rotation()).Inverse() * tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::LocalToGlobalDirection(unsigned int id, const CLHEP::Hep3Vector dir) const
{
  return GetDetector(id)->rotation() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GlobalToLocalDirection(unsigned int id, const CLHEP::Hep3Vector dir) const
{
  return (GetDetector(id)->rotation()).Inverse() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GetDetTranslation(unsigned int id) const
{
  DetGeomDesc *gd = GetDetector(id);
  CLHEP::Hep3Vector tmp;
  tmp.setX((gd->translation()).x());
  tmp.setY((gd->translation()).y());
  tmp.setZ((gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

void TotemRPGeometry::GetReadoutDirection(unsigned int id, double &dx, double &dy) const
{
  CLHEP::Hep3Vector d = LocalToGlobalDirection(id, CLHEP::Hep3Vector(0., 1., 0.));
  dx = d.x();
  dy = d.y();
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector TotemRPGeometry::GetRPGlobalTranslation(int copy_no) const
{
  CLHEP::Hep3Vector tmp;
  DetGeomDesc * gd = GetRPDevice(copy_no);
  tmp.setX((gd->translation()).x());
  tmp.setY((gd->translation()).y());
  tmp.setZ((gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::HepRotation TotemRPGeometry::GetRPGlobalRotation(int copy_no) const
{
  double xx, xy, xz, yx, yy, yz, zx, zy, zz;
  GetRPDevice(copy_no)->rotation().GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
  CLHEP::HepRep3x3 rot_mat( xx, xy, xz, yx, yy, yz, zx, zy, zz);
  CLHEP::HepRotation rot(rot_mat);
  return rot;
}


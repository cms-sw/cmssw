/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kaspar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSDDDNames.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include <iostream>

using namespace std;

//----------------------------------------------------------------------------------------------------

char CTPPSGeometry::build(const DetGeomDesc *gD)
{
  // reset
  theMap.clear();
  theRomanPotMap.clear();
  stationsInArm.clear();
  rpsInStation.clear();
  detsInRP.clear();

  // propagate through the GeometricalDet structure and add all detectors to 'theMap'
  deque<const DetGeomDesc *> buffer;
  buffer.push_back(gD);
  while (buffer.size() > 0)
  {
    const DetGeomDesc *d = buffer.front();
    buffer.pop_front();

    // check if it is a sensor
    if (d->name().name().compare(DDD_TOTEM_RP_SENSOR_NAME) == 0
        || d->name().name().compare(DDD_CTPPS_DIAMONDS_SEGMENT_NAME) == 0
	    || d->name().name().compare(DDD_CTPPS_PIXELS_SENSOR_NAME) == 0 )
    {
	  addSensor(d->geographicalID(), d);
    }

    // check if it is a RP
    if (d->name().name().compare(DDD_TOTEM_RP_RP_NAME) == 0
        || d->name().name().compare(DDD_CTPPS_PIXELS_RP_NAME) == 0
        || d->name().name().compare(DDD_CTPPS_DIAMONDS_RP_NAME) == 0
      )
    {
      addRP(d->geographicalID(), d);
    }
    
    for (unsigned int i = 0; i < d->components().size(); i++)
      buffer.push_back(d->components()[i]);
  }

  // build sets
  for (auto it = theMap.begin(); it != theMap.end(); ++it)
  {
    const CTPPSDetId detId(it->first);
    const CTPPSDetId rpId = detId.getRPId();
    const CTPPSDetId stId = detId.getStationId();
    const CTPPSDetId armId = detId.getArmId();

    stationsInArm[armId].insert(armId);
    rpsInStation[stId].insert(rpId);
    detsInRP[rpId].insert(detId);
  }

  return 0;
}

//----------------------------------------------------------------------------------------------------

char CTPPSGeometry::addSensor(unsigned int id, const DetGeomDesc* &gD)
{
  if (theMap.find(id) != theMap.end())
    return 1;

  theMap[id] = gD;
  return 0;
}

//----------------------------------------------------------------------------------------------------

char CTPPSGeometry::addRP(unsigned int id, const DetGeomDesc* &gD)
{
  if (theRomanPotMap.find(id) != theRomanPotMap.end())
    return 1;

  theRomanPotMap[id] = (DetGeomDesc*) gD;
  return 0;
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc* CTPPSGeometry::getSensor(unsigned int id) const
{
  auto it = theMap.find(id);
  if (it == theMap.end())
    throw cms::Exception("CTPPSGeometry") << "Not found detector with ID " << id << ", i.e. "
      << CTPPSDetId(id);

  return it->second;
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc* CTPPSGeometry::getRP(unsigned int id) const
{
  auto it = theRomanPotMap.find(id);
  if (it == theRomanPotMap.end())
    throw cms::Exception("CTPPSGeometry") << "Not found RP device with ID " << id << ", i.e. "
      << CTPPSDetId(id);

  return it->second;
}

//----------------------------------------------------------------------------------------------------
std::set<unsigned int> const& CTPPSGeometry::getStationsInArm(unsigned int id) const
{
  auto it = stationsInArm.find(id);
  if (it == stationsInArm.end())
    throw cms::Exception("CTPPSGeometry") << "Arm with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int> const& CTPPSGeometry::getRPsInStation(unsigned int id) const
{
  auto it = rpsInStation.find(id);
  if (it == rpsInStation.end())
    throw cms::Exception("CTPPSGeometry") << "Station with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int> const& CTPPSGeometry::getSensorsInRP(unsigned int id) const
{
  auto it = detsInRP.find(id);
  if (it == detsInRP.end())
    throw cms::Exception("CTPPSGeometry") << "RP with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::localToGlobal(const DetGeomDesc *gd, const CLHEP::Hep3Vector& r) const
{
  CLHEP::Hep3Vector tmp = gd->rotation() * r;
  tmp.setX(tmp.x() + (gd->translation()).x());
  tmp.setY(tmp.y() + (gd->translation()).y());
  tmp.setZ(tmp.z() + (gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::localToGlobal(unsigned int id, const CLHEP::Hep3Vector& r) const
{
  return localToGlobal(getSensor(id), r);
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::globalToLocal(const DetGeomDesc *gd, const CLHEP::Hep3Vector& r) const
{
  CLHEP::Hep3Vector tmp = r;
  tmp.setX(tmp.x() - (gd->translation()).x());
  tmp.setY(tmp.y() - (gd->translation()).y());
  tmp.setZ(tmp.z() - (gd->translation()).z());
  return (gd->rotation()).Inverse() * tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::globalToLocal(unsigned int id, const CLHEP::Hep3Vector& r) const
{
  return globalToLocal(getSensor(id), r);
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::localToGlobalDirection(unsigned int id, const CLHEP::Hep3Vector& dir) const
{
  return getSensor(id)->rotation() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::globalToLocalDirection(unsigned int id, const CLHEP::Hep3Vector& dir) const
{
  return (getSensor(id)->rotation()).Inverse() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::getSensorTranslation(unsigned int id) const
{
  auto gd = getSensor(id);

  CLHEP::Hep3Vector tmp;
  tmp.setX((gd->translation()).x());
  tmp.setY((gd->translation()).y());
  tmp.setZ((gd->translation()).z());

  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector CTPPSGeometry::getRPTranslation(unsigned int id) const
{
  auto gd = getRP(id);

  CLHEP::Hep3Vector tmp;
  tmp.setX((gd->translation()).x());
  tmp.setY((gd->translation()).y());
  tmp.setZ((gd->translation()).z());

  return tmp;
}

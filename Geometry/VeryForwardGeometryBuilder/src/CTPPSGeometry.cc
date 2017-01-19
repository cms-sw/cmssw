/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kaspar (jan.kaspar@gmail.com)
*  Laurent Forthomme (laurent.forthomme@cern.ch)
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DDDTotemRPCommon.h"
#include <iostream>

using namespace std;

//----------------------------------------------------------------------------------------------------

char
CTPPSGeometry::AddDetector( unsigned int id, const DetGeomDesc* &geom_desc )
{
  // check if the ID is already in map
  if ( theMap.find(id) != theMap.end() )
    return 1;

  // add geometry description
  theMap[id] = ( DetGeomDesc* )geom_desc;
  return 0;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc*
CTPPSGeometry::GetDetector( unsigned int id ) const
{
  // check if id is RP id?

  // check if there is a corresponding key
//  std::cout<<"TotemRPGeometry::GetDetector entered, id="<<id<<std::endl;
  mapType::const_iterator it = theMap.find( id );
  if ( it == theMap.end() )
    throw cms::Exception("TotemRPGeometry") << "Not found detector with ID " << id << ", i.e. "
      << CTPPSDetId(id);

  // the [] operator cannot be used as this method is const
  // and it must be const and one gets TotemRPGeometry const
  // from EventSetup
  LogDebug("CTPPSGeometry") << "Retrieved detector: " << id;
  return it->second;
}

//----------------------------------------------------------------------------------------------------

char
CTPPSGeometry::AddRPDevice(unsigned int id, const DetGeomDesc* &gD)
{
  // check if the copy_no is already in map
  if ( theRomanPotMap.find(id) != theRomanPotMap.end() )
    return 1;

  // add gD
  theRomanPotMap[id] = ( DetGeomDesc* )gD;
  return 0;
}

//----------------------------------------------------------------------------------------------------

DetGeomDesc*
CTPPSGeometry::GetRPDevice( unsigned int id ) const
{
  // check if there is a corresponding key
  RPDeviceMapType::const_iterator it = theRomanPotMap.find( id );
  if ( it == theRomanPotMap.end() )
    throw cms::Exception("TotemRPGeometry") << "Not found RP device with ID " << id << ", i.e. "
      << CTPPSDetId(id);

  return it->second;
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometry::BuildSets()
{
  // reset
  stationsInArm.clear();
  rpsInStation.clear();
  detsInRP.clear();

  // build
  for ( mapType::const_iterator it = theMap.begin(); it != theMap.end(); ++it )
  {
    const CTPPSDetId detId( it->first ), //FIXME TotemRPDetId?
                     rpId = detId.getRPId(),
                     stId = detId.getStationId(),
                     armId = detId.getArmId();

    stationsInArm[armId].insert( armId );
    rpsInStation[stId].insert( rpId );
    detsInRP[rpId].insert( detId );
  }
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int>
CTPPSGeometry::StationsInArm( unsigned int id ) const
{
  mapSetType::const_iterator it = stationsInArm.find( id );
  if ( it == stationsInArm.end() )
    throw cms::Exception("TotemRPGeometry") << "Arm with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int>
CTPPSGeometry::RPsInStation( unsigned int id ) const
{
  mapSetType::const_iterator it = rpsInStation.find( id );
  if ( it == rpsInStation.end() )
    throw cms::Exception("TotemRPGeometry") << "Station with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

std::set<unsigned int>
CTPPSGeometry::DetsInRP( unsigned int id ) const
{
  mapSetType::const_iterator it = detsInRP.find( id );
  if ( it == detsInRP.end() )
    throw cms::Exception("TotemRPGeometry") << "RP with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::LocalToGlobal( DetGeomDesc *gd, const CLHEP::Hep3Vector r ) const
{
  CLHEP::Hep3Vector tmp = gd->rotation() * r;
  tmp.setX(tmp.x() + (gd->translation()).x());
  tmp.setY(tmp.y() + (gd->translation()).y());
  tmp.setZ(tmp.z() + (gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::LocalToGlobal( unsigned int id, const CLHEP::Hep3Vector r ) const
{
  DetGeomDesc *gd = GetDetector( id );
  return LocalToGlobal( gd, r );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::GlobalToLocal( DetGeomDesc *gd, const CLHEP::Hep3Vector r ) const
{
  CLHEP::Hep3Vector tmp = r;
  tmp.setX(tmp.x() - (gd->translation()).x());
  tmp.setY(tmp.y() - (gd->translation()).y());
  tmp.setZ(tmp.z() - (gd->translation()).z());
  return (gd->rotation()).Inverse() * tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::GlobalToLocal( unsigned int id, const CLHEP::Hep3Vector r ) const
{
  DetGeomDesc *gd = GetDetector(id);
  return GlobalToLocal( gd, r );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::LocalToGlobalDirection( unsigned int id, const CLHEP::Hep3Vector dir ) const
{
  return GetDetector( id )->rotation() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::GlobalToLocalDirection( unsigned int id, const CLHEP::Hep3Vector dir ) const
{
  return GetDetector( id )->rotation().Inverse() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::GetDetTranslation( unsigned int id ) const
{
  DetGeomDesc *gd = GetDetector(id);
  CLHEP::Hep3Vector tmp;
  tmp.setX((gd->translation()).x());
  tmp.setY((gd->translation()).y());
  tmp.setZ((gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometry::GetReadoutDirection( unsigned int id, double &dx, double &dy ) const
{
  CLHEP::Hep3Vector d = LocalToGlobalDirection(id, CLHEP::Hep3Vector(0., 1., 0.));
  dx = d.x();
  dy = d.y();
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::GetRPGlobalTranslation( int copy_no ) const
{
  CLHEP::Hep3Vector tmp;
  DetGeomDesc * gd = GetRPDevice(copy_no);
  tmp.setX((gd->translation()).x());
  tmp.setY((gd->translation()).y());
  tmp.setZ((gd->translation()).z());
  return tmp;
}

//----------------------------------------------------------------------------------------------------

CLHEP::HepRotation
CTPPSGeometry::GetRPGlobalRotation( int copy_no ) const
{
  double xx, xy, xz, yx, yy, yz, zx, zy, zz;
  GetRPDevice(copy_no)->rotation().GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
  CLHEP::HepRep3x3 rot_mat( xx, xy, xz, yx, yy, yz, zx, zy, zz);
  CLHEP::HepRotation rot(rot_mat);
  return rot;
}


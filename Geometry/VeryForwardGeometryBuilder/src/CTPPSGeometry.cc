/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*  Jan Kaspar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include <regex>

//----------------------------------------------------------------------------------------------------

void
CTPPSGeometry::build( const DetGeomDesc* gD )
{
  // reset
  sensors_map_.clear();
  rps_map_.clear();
  stations_in_arm_.clear();
  rps_in_station_.clear();
  dets_in_rp_.clear();

  // propagate through the GeometricalDet structure and add all detectors to 'sensors_map_'
  std::deque<const DetGeomDesc *> buffer;
  buffer.emplace_back(gD);
  while ( !buffer.empty() ) {
    const DetGeomDesc *d = buffer.front();
    buffer.pop_front();

    // check if it is a sensor
    if ( d->name() == DDD_TOTEM_RP_SENSOR_NAME
      || std::regex_match( d->name(), std::regex( DDD_TOTEM_TIMING_SENSOR_TMPL ) )
      || d->name() == DDD_CTPPS_DIAMONDS_SEGMENT_NAME
      || d->name() == DDD_CTPPS_UFSD_SEGMENT_NAME
      || d->name() == DDD_CTPPS_PIXELS_SENSOR_NAME )
      addSensor( d->geographicalID(), d );

    // check if it is a RP
    if ( d->name() == DDD_TOTEM_RP_RP_NAME
      || d->name() == DDD_TOTEM_TIMING_RP_NAME
      || d->name() == DDD_CTPPS_DIAMONDS_RP_NAME
      || d->name() == DDD_CTPPS_PIXELS_RP_NAME )
      addRP( d->geographicalID(), d );

    for ( const auto& comp : d->components() )
      buffer.emplace_back( comp );
  }

  // build sets
  for ( const auto& it : sensors_map_ ) {
    const CTPPSDetId detId( it.first );
    const CTPPSDetId rpId = detId.getRPId();
    const CTPPSDetId stId = detId.getStationId();
    const CTPPSDetId armId = detId.getArmId();

    stations_in_arm_[armId].insert( armId );
    rps_in_station_[stId].insert( rpId );
    dets_in_rp_[rpId].insert( detId );
  }
}

//----------------------------------------------------------------------------------------------------

bool
CTPPSGeometry::addSensor( unsigned int id, const DetGeomDesc*& gD )
{
  if ( sensors_map_.find( id ) != sensors_map_.end() ) return false;

  sensors_map_[id] = gD;
  return true;
}

//----------------------------------------------------------------------------------------------------

bool
CTPPSGeometry::addRP( unsigned int id, const DetGeomDesc*& gD )
{
  if ( rps_map_.find( id ) != rps_map_.end() ) return false;

  rps_map_[id] = const_cast<DetGeomDesc*>( gD );
  return true;
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc*
CTPPSGeometry::getSensor( unsigned int id ) const
{
  auto g = getSensorNoThrow(id);
  if(nullptr ==g) {
    throw cms::Exception("CTPPSGeometry") << "Not found detector with ID " << id << ", i.e. "
      << CTPPSDetId( id );
  }
  return g;
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc*
CTPPSGeometry::getSensorNoThrow( unsigned int id ) const noexcept
{
  auto it = sensors_map_.find( id );
  if ( it == sensors_map_.end() ) {
    return nullptr;
  }
  return it->second;
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc*
CTPPSGeometry::getRP( unsigned int id ) const
{
  auto rp = getRPNoThrow(id);
  if(nullptr == rp) {     
    throw cms::Exception("CTPPSGeometry") << "Not found RP device with ID " << id << ", i.e. "
      << CTPPSDetId( id );
  }
  return rp;
}

//----------------------------------------------------------------------------------------------------

const DetGeomDesc*
CTPPSGeometry::getRPNoThrow( unsigned int id ) const noexcept
{
  auto it = rps_map_.find( id );
  if ( it == rps_map_.end() ) {
    return nullptr;
  }

  return it->second;
}

//----------------------------------------------------------------------------------------------------
const std::set<unsigned int>&
CTPPSGeometry::getStationsInArm( unsigned int id ) const
{
  auto it = stations_in_arm_.find( id );
  if ( it == stations_in_arm_.end() )
    throw cms::Exception("CTPPSGeometry") << "Arm with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

const std::set<unsigned int>&
CTPPSGeometry::getRPsInStation( unsigned int id ) const
{
  auto it = rps_in_station_.find( id );
  if ( it == rps_in_station_.end() )
    throw cms::Exception("CTPPSGeometry") << "Station with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

const std::set<unsigned int>&
CTPPSGeometry::getSensorsInRP( unsigned int id ) const
{
  auto it = dets_in_rp_.find( id );
  if ( it == dets_in_rp_.end() )
    throw cms::Exception("CTPPSGeometry") << "RP with ID " << id << " not found.";
  return it->second;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::localToGlobal( const DetGeomDesc* gd, const CLHEP::Hep3Vector& r ) const
{
  return gd->rotation() * r + CLHEP::Hep3Vector( gd->translation().x(), gd->translation().y(), gd->translation().z() );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::localToGlobal( unsigned int id, const CLHEP::Hep3Vector& r ) const
{
  return localToGlobal( getSensor( id ), r );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::globalToLocal( const DetGeomDesc* gd, const CLHEP::Hep3Vector& r ) const
{
  return gd->rotation().Inverse() * ( r - CLHEP::Hep3Vector( gd->translation().x(), gd->translation().y(), gd->translation().z() ) );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::globalToLocal( unsigned int id, const CLHEP::Hep3Vector& r ) const
{
  return globalToLocal( getSensor( id ), r );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::localToGlobalDirection( unsigned int id, const CLHEP::Hep3Vector& dir ) const
{
  return getSensor( id )->rotation() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::globalToLocalDirection( unsigned int id, const CLHEP::Hep3Vector& dir ) const
{
  return getSensor( id )->rotation().Inverse() * dir;
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::getSensorTranslation( unsigned int id ) const
{
  auto gd = getSensor( id );
  return CLHEP::Hep3Vector( gd->translation().x(), gd->translation().y(), gd->translation().z() );
}

//----------------------------------------------------------------------------------------------------

CLHEP::Hep3Vector
CTPPSGeometry::getRPTranslation( unsigned int id ) const
{
  auto gd = getRP( id );
  return CLHEP::Hep3Vector( gd->translation().x(), gd->translation().y(), gd->translation().z() );
}

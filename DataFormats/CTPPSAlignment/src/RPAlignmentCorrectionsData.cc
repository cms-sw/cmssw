/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

#include "FWCore/Utilities/interface/typelookup.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

#include <set>

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData&
RPAlignmentCorrectionsData::getRPCorrection( unsigned int id )
{
  return rps_[id];
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData
RPAlignmentCorrectionsData::getRPCorrection( unsigned int id ) const
{
  RPAlignmentCorrectionData align_corr;
  auto it = rps_.find( id );
  if ( it != rps_.end() )
    align_corr = it->second;
  return align_corr;
} 

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData&
RPAlignmentCorrectionsData::getSensorCorrection( unsigned int id )
{
  return sensors_[id];
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData
RPAlignmentCorrectionsData::getSensorCorrection( unsigned int id ) const
{
  RPAlignmentCorrectionData align_corr;
  auto it = sensors_.find( id );
  if ( it != sensors_.end() )
    align_corr = it->second;
  return align_corr;
}

//----------------------------------------------------------------------------------------------------

RPAlignmentCorrectionData
RPAlignmentCorrectionsData::getFullSensorCorrection( unsigned int id, bool useRPErrors ) const
{
  RPAlignmentCorrectionData align_corr;

  // try to get alignment correction of the full RP
  auto rpIt = rps_.find( CTPPSDetId( id ).getRPId() );
  if ( rpIt != rps_.end() )
    align_corr = rpIt->second;

  // try to get sensor alignment correction
  auto sIt = sensors_.find( id );

  // merge the corrections
  if ( sIt != sensors_.end() )
    align_corr.add( sIt->second, useRPErrors );

  return align_corr;
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsData::setRPCorrection( unsigned int id, const RPAlignmentCorrectionData& ac )
{
  rps_[id] = ac;
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsData::setSensorCorrection( unsigned int id, const RPAlignmentCorrectionData& ac )
{
  sensors_[id] = ac;
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsData::addRPCorrection( unsigned int id, const RPAlignmentCorrectionData &a, bool sumErrors, bool addShR, bool addShZ, bool addRotZ )
{
  auto it = rps_.find( id );
  if ( it == rps_.end() )
    rps_.insert( mapType::value_type( id, a ) );
  else
    it->second.add( a, sumErrors, addShR, addShZ, addRotZ );
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsData::addSensorCorrection( unsigned int id, const RPAlignmentCorrectionData &a, bool sumErrors, bool addShR, bool addShZ, bool addRotZ )
{
  auto it = sensors_.find( id );
  if ( it == sensors_.end() )
    sensors_.insert( mapType::value_type( id, a ) );
  else
    it->second.add( a, sumErrors, addShR, addShZ, addRotZ );
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsData::addCorrections( const RPAlignmentCorrectionsData &nac, bool sumErrors, bool addShR, bool addShZ, bool addRotZ )
{
  for ( const auto& it : nac.rps_ )
    addRPCorrection( it.first, it.second, sumErrors, addShR, addShZ, addRotZ );

  for ( const auto& it : nac.sensors_ )
    addSensorCorrection( it.first, it.second, sumErrors, addShR, addShZ, addRotZ );
}

//----------------------------------------------------------------------------------------------------

void
RPAlignmentCorrectionsData::clear()
{
  rps_.clear();
  sensors_.clear();
}

//----------------------------------------------------------------------------------------------------

TYPELOOKUP_DATA_REG( RPAlignmentCorrectionsData );

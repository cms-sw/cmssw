/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Filip Dej
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"

//--------------------------------------------------------------------------

bool
PPSTimingCalibration::CalibrationKey::operator<( const PPSTimingCalibration::CalibrationKey& rhs ) const
{
  if ( db == rhs.db ) {
    if ( sampic == rhs.sampic ) {
      if ( channel == rhs.channel )
        return cell < rhs.cell;
      return channel < rhs.channel;
    }
    return sampic < rhs.sampic;
  }
  return db < rhs.db;
}

std::ostream&
operator<<( std::ostream& os, const PPSTimingCalibration::CalibrationKey& key )
{
  return os << key.db << " " << key.sampic << " " << key.channel << " " << key.cell;
}

//--------------------------------------------------------------------------

std::vector<double>
PPSTimingCalibration::getParameters( int db, int sampic, int channel, int cell ) const
{
  CalibrationKey key( db, sampic, channel, cell );
  auto out = parameters_.find( key );
  if ( out == parameters_.end() )
    return {};
  return out->second;
}

double
PPSTimingCalibration::getTimeOffset( int db, int sampic, int channel ) const
{
  CalibrationKey key( db, sampic, channel );
  auto out = timeInfo_.find( key );
  if ( out == timeInfo_.end() )
    return 0.;
  return out->second.first;
}

double
PPSTimingCalibration::getTimePrecision( int db, int sampic, int channel ) const
{
  CalibrationKey key( db, sampic, channel );
  auto out = timeInfo_.find( key );
  if ( out == timeInfo_.end() )
    return 0.;
  return out->second.second;
}

std::ostream&
operator<<( std::ostream& os, const PPSTimingCalibration& data )
{
  os << "FORMULA: "<< data.formula_ << "\nDB SAMPIC CHANNEL CELL PARAMETERS TIME_OFFSET\n";
  for ( const auto& kv : data.parameters_ ) {
    os << kv.first <<" [";
    for ( size_t i = 0; i < kv.second.size(); ++i )
      os << ( i > 0 ? ", " : "" ) << kv.second.at( i );
    PPSTimingCalibration::CalibrationKey k = kv.first;
    k.cell = -1;
    os << "] " << data.timeInfo_.at( k ).first << " " <<  data.timeInfo_.at( k ).second << "\n";
  }
  return os;
}


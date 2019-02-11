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
PPSTimingCalibration::Key::operator<( const PPSTimingCalibration::Key& rhs ) const
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
operator<<( std::ostream& os, const PPSTimingCalibration::Key& key )
{
  return os << key.db << " " << key.sampic << " " << key.channel << " " << key.cell;
}

//--------------------------------------------------------------------------

std::vector<double>
PPSTimingCalibration::parameters( int db, int sampic, int channel, int cell ) const
{
  Key key{ db, sampic, channel, cell };
  auto out = parameters_.find( key );
  if ( out == parameters_.end() )
    return {};
  return out->second;
}

double
PPSTimingCalibration::timeOffset( int db, int sampic, int channel ) const
{
  Key key{ db, sampic, channel, -1 };
  auto out = timeInfo_.find( key );
  if ( out == timeInfo_.end() )
    return 0.;
  return out->second.first;
}

double
PPSTimingCalibration::timePrecision( int db, int sampic, int channel ) const
{
  Key key{ db, sampic, channel, -1 };
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
    PPSTimingCalibration::Key k = kv.first;
    k.cell = -1;
    const auto& time = data.timeInfo_.at( k );
    os << "] " << time.first << " " <<  time.second << "\n";
  }
  return os;
}


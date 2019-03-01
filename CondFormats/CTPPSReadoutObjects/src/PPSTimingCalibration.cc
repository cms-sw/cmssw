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
  if ( key1 == rhs.key1 ) {
    if ( key2 == rhs.key2 ) {
      if ( key3 == rhs.key3 )
        return key4 < rhs.key4;
      return key3 < rhs.key4;
    }
    return key2 < rhs.key2;
  }
  return key1 < rhs.key1;
}

std::ostream&
operator<<( std::ostream& os, const PPSTimingCalibration::Key& key )
{
  return os << key.key1 << " " << key.key2 << " " << key.key3 << " " << key.key4;
}

//--------------------------------------------------------------------------

std::vector<double>
PPSTimingCalibration::parameters( int key1, int key2, int key3, int key4 ) const
{
  Key key{ key1, key2, key3, key4 };
  auto out = parameters_.find( key );
  if ( out == parameters_.end() )
    return {};
  return out->second;
}

double
PPSTimingCalibration::timeOffset( int key1, int key2, int key3, int key4 ) const
{
  Key key{ key1, key2, key3, key4 };
  auto out = timeInfo_.find( key );
  if ( out == timeInfo_.end() )
    return 0.;
  return out->second.first;
}

double
PPSTimingCalibration::timePrecision( int key1, int key2, int key3, int key4 ) const
{
  Key key{ key1, key2, key3, key4 };
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
    const auto& time = data.timeInfo_.at( k );
    os << "] " << time.first << " " <<  time.second << "\n";
  }
  return os;
}


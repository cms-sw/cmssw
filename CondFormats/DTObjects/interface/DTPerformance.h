#ifndef DTPerformance_H
#define DTPerformance_H
/** \class DTPerformance
 *
 *  Description:
 *       Class to hold drift tubes performances ( SL by SL )
 *
 *  $Date: 2006/06/28 17:00:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/DTObjects/interface/DTTimeUnits.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTPerformanceId {

 public:

  DTPerformanceId();
  ~DTPerformanceId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;

};


class DTPerformanceData {

 public:

  DTPerformanceData();
  ~DTPerformanceData();

  float meanT0;
  float meanTtrig;
  float meanMtime;
  float meanNoise;
  float meanAfterPulse;
  float meanResolution;
  float meanEfficiency;

};


class DTPerformanceCompare {
 public:
  bool operator()( const DTPerformanceId& idl,
                   const DTPerformanceId& idr ) const;
};


class DTPerformance {

 public:

  /** Constructor
   */
  DTPerformance();
  DTPerformance( const std::string& version );

  /** Destructor
   */
  ~DTPerformance();

  /** Operations
   */
  /// get content
  int slPerformance( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     float& meanT0,
                     float& meanTtrig,
                     float& meanMtime,
                     float& meanNoise,
                     float& meanAfterPulse,
                     float& meanResolution,
                     float& meanEfficiency,
                     DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int slPerformance( const DTSuperLayerId& id,
                     float& meanT0,
                     float& meanTtrig,
                     float& meanMtime,
                     float& meanNoise,
                     float& meanAfterPulse,
                     float& meanResolution,
                     float& meanEfficiency,
                     DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setSLPerformance( int   wheelId,
                        int stationId,
                        int  sectorId,
                        int      slId,
                        float meanT0,
                        float meanTtrig,
                        float meanMtime,
                        float meanNoise,
                        float meanAfterPulse,
                        float meanResolution,
                        float meanEfficiency,
                        DTTimeUnits::type unit = DTTimeUnits::counts );
  int setSLPerformance( const DTSuperLayerId& id,
                        float meanT0,
                        float meanTtrig,
                        float meanMtime,
                        float meanNoise,
                        float meanAfterPulse,
                        float meanResolution,
                        float meanEfficiency,
                        DTTimeUnits::type unit = DTTimeUnits::counts );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::map<DTPerformanceId,
                   DTPerformanceData,
                   DTPerformanceCompare>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::map<DTPerformanceId,DTPerformanceData,DTPerformanceCompare> slData;

};


#endif // DTPerformance_H


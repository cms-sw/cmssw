#ifndef DTPerformance_H
#define DTPerformance_H
/** \class DTPerformance
 *
 *  Description:
 *       Class to hold drift tubes performances ( SL by SL )
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/DTObjects/interface/DTTimeUnits.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "FWCore/Utilities/interface/ConstRespectingPtr.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <utility>

template <class Key, class Content> class DTBufferTree;

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


 COND_SERIALIZABLE;
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


 COND_SERIALIZABLE;
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
                     DTTimeUnits::type unit ) const
      { return get( wheelId, stationId, sectorId, slId,
                    meanT0, meanTtrig, meanMtime, meanNoise, meanAfterPulse, 
                    meanResolution, meanEfficiency, unit ); };
  int slPerformance( const DTSuperLayerId& id,
                     float& meanT0,
                     float& meanTtrig,
                     float& meanMtime,
                     float& meanNoise,
                     float& meanAfterPulse,
                     float& meanResolution,
                     float& meanEfficiency,
                     DTTimeUnits::type unit ) const
      { return get( id,
                    meanT0, meanTtrig, meanMtime, meanNoise, meanAfterPulse,
                    meanResolution, meanEfficiency, unit ); };
  int get( int   wheelId,
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
           DTTimeUnits::type unit ) const;
  int get( const DTSuperLayerId& id,
           float& meanT0,
           float& meanTtrig,
           float& meanMtime,
           float& meanNoise,
           float& meanAfterPulse,
           float& meanResolution,
           float& meanEfficiency,
           DTTimeUnits::type unit ) const;
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
                        DTTimeUnits::type unit )
      { return set( wheelId, stationId, sectorId, slId,
                    meanT0, meanTtrig, meanMtime, meanNoise, meanAfterPulse,
                    meanResolution, meanEfficiency, unit ); };
  int setSLPerformance( const DTSuperLayerId& id,
                        float meanT0,
                        float meanTtrig,
                        float meanMtime,
                        float meanNoise,
                        float meanAfterPulse,
                        float meanResolution,
                        float meanEfficiency,
                        DTTimeUnits::type unit )
      { return set( id,
                    meanT0, meanTtrig, meanMtime, meanNoise, meanAfterPulse,
                    meanResolution, meanEfficiency, unit ); };
  int set( int   wheelId,
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
           DTTimeUnits::type unit );
  int set( const DTSuperLayerId& id,
           float meanT0,
           float meanTtrig,
           float meanMtime,
           float meanNoise,
           float meanAfterPulse,
           float meanResolution,
           float meanEfficiency,
           DTTimeUnits::type unit );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::vector< std::pair<DTPerformanceId,
                                 DTPerformanceData> >::const_iterator
                                                       const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  void initialize();

 private:

  DTPerformance(DTPerformance const&) = delete;
  DTPerformance& operator=(DTPerformance const&) = delete;

  std::string dataVersion;
  float nsPerCount;

  std::vector< std::pair<DTPerformanceId,DTPerformanceData> > dataList;

  DTBufferTree<int,int>* dBuf COND_TRANSIENT;

  /// read and store full content
  std::string mapName() const;


 COND_SERIALIZABLE;
};
#endif // DTPerformance_H

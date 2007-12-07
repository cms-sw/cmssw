#ifndef DTMtime_H
#define DTMtime_H
/** \class DTMtime
 *
 *  Description:
 *       Class to hold drift tubes mean-times
 *             ( SL by SL mean-time calculation )
 *
 *  $Date: 2007/10/30 17:30:20 $
 *  $Revision: 1.6.6.1 $
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
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTMtimeId {

 public:

  DTMtimeId();
  ~DTMtimeId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

};


class DTMtimeData {

 public:

  DTMtimeData();
  ~DTMtimeData();

  float mTime;
  float mTrms;

};


class DTMtime {

 public:

  /** Constructor
   */
  DTMtime();
  DTMtime( const std::string& version );

  /** Destructor
   */
  ~DTMtime();

  /** Operations
   */
  /// get content
  int slMtime( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               float&  mTime,
               float&  mTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( wheelId, stationId, sectorId, slId, 0, 0,
                    mTime, mTrms, unit ); };
  int slMtime( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int   layerId,
               int    cellId,
               float&  mTime,
               float&  mTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( wheelId, stationId, sectorId, slId, layerId, cellId,
                    mTime, mTrms, unit ); };
  int slMtime( const DTSuperLayerId& id,
               float&  mTime,
               float&  mTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( id, mTime, mTrms, unit ); };
  int slMtime( const DetId& id,
               float&  mTime,
               float&  mTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( id, mTime, mTrms, unit ); };
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           float&  mTime,
           float&  mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           float&  mTime,
           float&  mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int get( const DTSuperLayerId& id,
           float&  mTime,
           float&  mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int get( const DetId& id,
           float&  mTime,
           float&  mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setSLMtime( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( wheelId, stationId, sectorId, slId, 0, 0,
                    mTime, mTrms, unit ); };
  int setSLMtime( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( wheelId, stationId, sectorId, slId, layerId, cellId,
                    mTime, mTrms, unit ); };
  int setSLMtime( const DTSuperLayerId& id,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( id, mTime, mTrms, unit ); };
  int setSLMtime( const DetId& id,
                  float   mTime,
                  float   mTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( id, mTime, mTrms, unit ); };
  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           float   mTime,
           float   mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           float   mTime,
           float   mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  int set( const DTSuperLayerId& id,
           float   mTime,
           float   mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  int set( const DetId& id,
           float   mTime,
           float   mTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::vector< std::pair<DTMtimeId,
                                 DTMtimeData> >::const_iterator
                                                 const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::vector< std::pair<DTMtimeId,DTMtimeData> > dataList;

  /// read and store full content
  void cacheMap() const;
  std::string mapName() const;

};


#endif // DTMtime_H


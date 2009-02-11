#ifndef DTTtrig_H
#define DTTtrig_H
/** \class DTTtrig
 *
 *  Description:
 *       Class to hold drift tubes TTrigs
 *             ( SL by SL time offsets )
 *
 *  $Date: 2007/10/30 17:30:20 $
 *  $Revision: 1.5.6.1 $
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

class DTTtrigId   {

 public:

  DTTtrigId();
  ~DTTtrigId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

};


class DTTtrigData {

 public:

  DTTtrigData();
  ~DTTtrigData();

  float tTrig;
  float tTrms;

};


class DTTtrig {

 public:

  /** Constructor
   */
  DTTtrig();
  DTTtrig( const std::string& version );

  /** Destructor
   */
  ~DTTtrig();

  /** Operations
   */
  /// get content
  int slTtrig( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               float&  tTrig,
               float&  tTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( wheelId, stationId, sectorId, slId, 0, 0,
                    tTrig, tTrms, unit ); };
  int slTtrig( int   wheelId,
               int stationId,
               int  sectorId,
               int      slId,
               int   layerId,
               int    cellId,
               float&  tTrig,
               float&  tTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( wheelId, stationId, sectorId, slId, layerId, cellId,
                    tTrig, tTrms, unit ); };
  int slTtrig( const DTSuperLayerId& id,
               float&  tTrig,
               float&  tTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( id, tTrig, tTrms, unit ); };
  int slTtrig( const DetId& id,
               float&  tTrig,
               float&  tTrms,
               DTTimeUnits::type unit = DTTimeUnits::counts ) const
      { return get( id, tTrig, tTrms, unit ); };
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           float&  tTrig,
           float&  tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           float&  tTrig,
           float&  tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int get( const DTSuperLayerId& id,
           float&  tTrig,
           float&  tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int get( const DetId& id,
           float&  tTrig,
           float&  tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setSLTtrig( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( wheelId, stationId, sectorId, slId, 0, 0,
                    tTrig, tTrms, unit ); };
  int setSLTtrig( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( wheelId, stationId, sectorId, slId, layerId, cellId,
                    tTrig, tTrms, unit ); };
  int setSLTtrig( const DTSuperLayerId& id,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( id, tTrig, tTrms, unit ); };
  int setSLTtrig( const DetId& id,
                  float   tTrig,
                  float   tTrms,
                  DTTimeUnits::type unit = DTTimeUnits::counts )
      { return set( id, tTrig, tTrms, unit ); };
  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           float   tTrig,
           float   tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           float   tTrig,
           float   tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  int set( const DTSuperLayerId& id,
           float   tTrig,
           float   tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  int set( const DetId& id,
           float   tTrig,
           float   tTrms,
           DTTimeUnits::type unit = DTTimeUnits::counts );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::vector< std::pair<DTTtrigId,
                                 DTTtrigData> >::const_iterator
                                                 const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::vector< std::pair<DTTtrigId,DTTtrigData> > dataList;

  /// read and store full content
  void cacheMap() const;
  std::string mapName() const;

};


#endif // DTTtrig_H


#ifndef DTT0_H
#define DTT0_H
/** \class DTT0
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *             ( cell by cell time offsets )
 *
 *  $Date: 2008/09/29 13:10:34 $
 *  $Revision: 1.8 $
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
#include "CondFormats/DTObjects/interface/DTBufferTree.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTT0Id {

 public:

  DTT0Id();
  ~DTT0Id();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

};


class DTT0Data {

 public:

  DTT0Data();
  ~DTT0Data();

  float t0mean;
  float t0rms;

};


class DTT0 {

 public:

  /** Constructor
   */
  DTT0();
  DTT0( const std::string& version );

  /** Destructor
   */
  ~DTT0();

  /** Operations
   */
  /// get content
  int cellT0( int   wheelId,
              int stationId,
              int  sectorId,
              int      slId,
              int   layerId,
              int    cellId,
              float& t0mean,
              float& t0rms,
              DTTimeUnits::type unit ) const
      { return get( wheelId, stationId, sectorId, slId, layerId, cellId,
                    t0mean, t0rms, unit ); };
  int cellT0( const DTWireId& id,
              float& t0mean,
              float& t0rms,
              DTTimeUnits::type unit ) const
      { return get( id, t0mean, t0rms, unit ); };
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           float& t0mean,
           float& t0rms,
           DTTimeUnits::type unit ) const;
  int get( const DTWireId& id,
           float& t0mean,
           float& t0rms,
           DTTimeUnits::type unit ) const;
  float unit() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setCellT0( int   wheelId,
                 int stationId,
                 int  sectorId,
                 int      slId,
                 int   layerId,
                 int    cellId,
                 float t0mean,
                 float t0rms,
                 DTTimeUnits::type unit )
      { return set( wheelId, stationId, sectorId, slId, layerId, cellId,
                    t0mean, t0rms, unit ); };
  int setCellT0( const DTWireId& id,
                 float t0mean,
                 float t0rms,
                 DTTimeUnits::type unit )
      { return set( id, t0mean, t0rms, unit ); };
  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           float t0mean,
           float t0rms,
           DTTimeUnits::type unit );
  int set( const DTWireId& id,
           float t0mean,
           float t0rms,
           DTTimeUnits::type unit );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::vector< std::pair<DTT0Id,
                                 DTT0Data> >::const_iterator
                                              const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::vector< std::pair<DTT0Id,DTT0Data> > dataList;

  DTBufferTree<int,int>* dBuf;

  /// read and store full content
  void cacheMap() const;
  std::string mapName() const;

};


#endif // DTT0_H


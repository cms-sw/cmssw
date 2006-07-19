#ifndef DTT0_H
#define DTT0_H
/** \class DTT0
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *             ( cell by cell time offsets )
 *
 *  $Date: 2006/06/29 14:20:02 $
 *  $Revision: 1.5 $
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

//---------------
// C++ Headers --
//---------------
#include <string>
#include <map>

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


class DTT0Compare {
 public:
  bool operator()( const DTT0Id& idl,
                   const DTT0Id& idr ) const;
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
              DTTimeUnits::type unit = DTTimeUnits::counts ) const;
  int cellT0( const DTWireId& id,
              float& t0mean,
              float& t0rms,
              DTTimeUnits::type unit = DTTimeUnits::counts ) const;
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
                 DTTimeUnits::type unit = DTTimeUnits::counts );
  int setCellT0( const DTWireId& id,
                 float t0mean,
                 float t0rms,
                 DTTimeUnits::type unit = DTTimeUnits::counts );
  void setUnit( float unit );

  /// Access methods to data
  typedef std::map<DTT0Id,
                   DTT0Data,
                   DTT0Compare>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;
  float nsPerCount;

  std::map<DTT0Id,DTT0Data,DTT0Compare> cellData;

};


#endif // DTT0_H


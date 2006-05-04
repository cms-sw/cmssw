#ifndef DTT0_H
#define DTT0_H
/** \class DTT0
 *
 *  Description:
 *       Class to hold drift tubes T0s
 *             ( cell by cell time offsets )
 *
 *  $Date: 2006/02/28 18:06:29 $
 *  $Revision: 1.4 $
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
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCellT0Data {

 public:

  DTCellT0Data();
  ~DTCellT0Data();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;
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
  typedef std::vector<DTCellT0Data>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  /// read and store full content
  void initSetup() const;

  std::string dataVersion;
  float nsPerCount;

  std::vector<DTCellT0Data> cellData;

};


#endif // DTT0_H


#ifndef DTStatusFlag_H
#define DTStatusFlag_H
/** \class DTStatusFlag
 *
 *  Description:
 *       Class to hold drift tubes status ( noise and masks )
 *             ( cell by cell time offsets )
 *
 *  $Date: 2006/05/16 12:00:00 $
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
#include "DataFormats/MuonDetId/interface/DTWireId.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCellStatusFlagData {

 public:

  DTCellStatusFlagData();
  ~DTCellStatusFlagData();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;
  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;

};


class DTStatusFlag {

 public:

  /** Constructor
   */
  DTStatusFlag();
  DTStatusFlag( const std::string& version );

  /** Destructor
   */
  ~DTStatusFlag();

  /** Operations
   */
  /// get content
  int cellStatus( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  bool& noiseFlag,
                  bool&    feMask,
                  bool&   tdcMask ) const;
  int cellStatus( const DTWireId& id,
                  bool& noiseFlag,
                  bool&    feMask,
                  bool&   tdcMask ) const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  int setCellStatus( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     bool noiseFlag,
                     bool    feMask,
                     bool   tdcMask );
  int setCellStatus( const DTWireId& id,
                     bool noiseFlag,
                     bool    feMask,
                     bool   tdcMask );

  /// Access methods to data
  typedef std::vector<DTCellStatusFlagData>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  /// read and store full content
  void initSetup() const;

  std::string dataVersion;

  std::vector<DTCellStatusFlagData> cellData;

};


#endif // DTStatusFlag_H


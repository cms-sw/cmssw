#ifndef DTStatusFlag_H
#define DTStatusFlag_H
/** \class DTStatusFlag
 *
 *  Description:
 *       Class to hold drift tubes status ( noise and masks )
 *             ( cell by cell time offsets )
 *
 *  $Date: 2006/05/17 10:33:51 $
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
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;

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
                  bool&   tdcMask,
                  bool&  trigMask,
                  bool&  deadFlag,
                  bool&  nohvFlag ) const;
  int cellStatus( const DTWireId& id,
                  bool& noiseFlag,
                  bool&    feMask,
                  bool&   tdcMask,
                  bool&  trigMask,
                  bool&  deadFlag,
                  bool&  nohvFlag ) const;

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
                     bool   tdcMask,
                     bool  trigMask,
                     bool  deadFlag,
                     bool  nohvFlag );
  int setCellStatus( const DTWireId& id,
                     bool noiseFlag,
                     bool    feMask,
                     bool   tdcMask,
                     bool  trigMask,
                     bool  deadFlag,
                     bool  nohvFlag );

  int setCellNoise( int   wheelId,
                    int stationId,
                    int  sectorId,
                    int      slId,
                    int   layerId,
                    int    cellId,
                    bool flag );
  int setCellNoise( const DTWireId& id,
                    bool flag );

  int setCellFEMask( int   wheelId,
                     int stationId,
                     int  sectorId,
                     int      slId,
                     int   layerId,
                     int    cellId,
                     bool mask );
  int setCellFEMask( const DTWireId& id,
                     bool mask );

  int setCellTDCMask( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int   layerId,
                      int    cellId,
                      bool mask );
  int setCellTDCMask( const DTWireId& id,
                      bool mask );

  int setCellTrigMask( int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId,
                       bool mask );
  int setCellTrigMask( const DTWireId& id,
                       bool mask );

  int setCellDead( int   wheelId,
                   int stationId,
                   int  sectorId,
                   int      slId,
                   int   layerId,
                   int    cellId,
                   bool flag );
  int setCellDead( const DTWireId& id,
                   bool flag );

  int setCellNoHV( int   wheelId,
                   int stationId,
                   int  sectorId,
                   int      slId,
                   int   layerId,
                   int    cellId,
                   bool flag );
  int setCellNoHV( const DTWireId& id,
                   bool flag );

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


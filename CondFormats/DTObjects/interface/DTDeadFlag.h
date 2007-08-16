#ifndef DTDeadFlag_H
#define DTDeadFlag_H
/** \class DTDeadFlag
 *
 *  Description:
 *       Class to hold drift tubes life and HV status
 *
 *  $Date: 2007/08/15 12:00:00 $
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
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDeadFlagId {

 public:

  DTDeadFlagId();
  ~DTDeadFlagId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

};


class DTDeadFlagData {

 public:

  DTDeadFlagData();
  ~DTDeadFlagData();

  bool deadFlag;
  bool nohvFlag;

};


class DTDeadFlagCompare {
 public:
  bool operator()( const DTDeadFlagId& idl,
                   const DTDeadFlagId& idr ) const;
};


class DTDeadFlag {

 public:

  /** Constructor
   */
  DTDeadFlag();
  DTDeadFlag( const std::string& version );

  /** Destructor
   */
  ~DTDeadFlag();

  /** Operations
   */
  /// get content
  int cellStatus( int   wheelId,
                  int stationId,
                  int  sectorId,
                  int      slId,
                  int   layerId,
                  int    cellId,
                  bool& deadFlag,
                  bool& nohvFlag ) const;
  int cellStatus( const DTWireId& id,
                  bool& deadFlag,
                  bool& nohvFlag ) const;

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
                     bool deadFlag,
                     bool nohvFlag );
  int setCellStatus( const DTWireId& id,
                     bool deadFlag,
                     bool nohvFlag );

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
  typedef std::map<DTDeadFlagId,
                   DTDeadFlagData,
                   DTDeadFlagCompare>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string dataVersion;

  std::map<DTDeadFlagId,DTDeadFlagData,DTDeadFlagCompare> cellData;

};


#endif // DTDeadFlag_H


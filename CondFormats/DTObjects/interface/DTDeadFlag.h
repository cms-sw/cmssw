#ifndef DTDeadFlag_H
#define DTDeadFlag_H
/** \class DTDeadFlag
 *
 *  Description:
 *       Class to hold drift tubes life and HV status
 *
 *  $Date: 2007/08/16 10:53:13 $
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

  bool dead_HV;
  bool dead_TP;
  bool dead_RO;
  bool discCat;

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
                  bool& dead_HV,
                  bool& dead_TP,
                  bool& dead_RO,
                  bool& discCat ) const;
  int cellStatus( const DTWireId& id,
                  bool& dead_HV,
                  bool& dead_TP,
                  bool& dead_RO,
                  bool& discCat ) const;

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
                     bool dead_HV,
                     bool dead_TP,
                     bool dead_RO,
                     bool discCat );
  int setCellStatus( const DTWireId& id,
                     bool dead_HV,
                     bool dead_TP,
                     bool dead_RO,
                     bool discCat );

  int setCellDead_HV( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int   layerId,
                      int    cellId,
                      bool flag );
  int setCellDead_HV( const DTWireId& id,
                      bool flag );

  int setCellDead_TP( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int   layerId,
                      int    cellId,
                      bool flag );
  int setCellDead_TP( const DTWireId& id,
                      bool flag );

  int setCellDead_RO( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int   layerId,
                      int    cellId,
                      bool flag );
  int setCellDead_RO( const DTWireId& id,
                      bool flag );

  int setCellDiscCat( int   wheelId,
                      int stationId,
                      int  sectorId,
                      int      slId,
                      int   layerId,
                      int    cellId,
                      bool flag );
  int setCellDiscCat( const DTWireId& id,
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


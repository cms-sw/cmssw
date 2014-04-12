#ifndef DTDeadFlag_H
#define DTDeadFlag_H
/** \class DTDeadFlag
 *
 *  Description:
 *       Class to hold drift tubes life and HV status
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
#include "DataFormats/MuonDetId/interface/DTWireId.h"
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
                  bool& discCat ) const
      { return get( wheelId, stationId, sectorId, slId, layerId, cellId, 
                    dead_HV, dead_TP, dead_RO, discCat ); };
  int cellStatus( const DTWireId& id,
                  bool& dead_HV,
                  bool& dead_TP,
                  bool& dead_RO,
                  bool& discCat ) const
      { return get( id, dead_HV, dead_TP, dead_RO, discCat ); };
  int get( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           bool& dead_HV,
           bool& dead_TP,
           bool& dead_RO,
           bool& discCat ) const;
  int get( const DTWireId& id,
           bool& dead_HV,
           bool& dead_TP,
           bool& dead_RO,
           bool& discCat ) const;

  bool getCellDead_HV( int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId ) const;
  bool getCellDead_HV( const DTWireId& id ) const;

  bool getCellDead_TP( int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId ) const;
  bool getCellDead_TP( const DTWireId& id ) const;

  bool getCellDead_RO( int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId ) const;
  bool getCellDead_RO( const DTWireId& id ) const;

  bool getCellDiscCat( int   wheelId,
                       int stationId,
                       int  sectorId,
                       int      slId,
                       int   layerId,
                       int    cellId ) const;
  bool getCellDiscCat( const DTWireId& id ) const;

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
                     bool discCat )
      { return set( wheelId, stationId, sectorId, slId, layerId, cellId, 
                    dead_HV, dead_TP, dead_RO, discCat ); };
  int setCellStatus( const DTWireId& id,
                     bool dead_HV,
                     bool dead_TP,
                     bool dead_RO,
                     bool discCat )
      { return set( id, dead_HV, dead_TP, dead_RO, discCat ); };

  int set( int   wheelId,
           int stationId,
           int  sectorId,
           int      slId,
           int   layerId,
           int    cellId,
           bool dead_HV,
           bool dead_TP,
           bool dead_RO,
           bool discCat ) ;
  int set( const DTWireId& id,
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
  typedef std::vector< std::pair<DTDeadFlagId,
                                 DTDeadFlagData> >::const_iterator
                                                    const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  void initialize();

 private:

  DTDeadFlag(DTDeadFlag const&);
  DTDeadFlag& operator=(DTDeadFlag const&);

  std::string dataVersion;

  std::vector< std::pair<DTDeadFlagId,DTDeadFlagData> > dataList;

  edm::ConstRespectingPtr<DTBufferTree<int,int> > dBuf;

  /// read and store full content
  std::string mapName() const;

};

#endif // DTDeadFlag_H

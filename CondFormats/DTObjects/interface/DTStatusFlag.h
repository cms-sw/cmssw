#ifndef DTStatusFlag_H
#define DTStatusFlag_H
/** \class DTStatusFlag
 *
 *  Description:
 *       Class to hold drift tubes status
 *             ( cell by cell noise and masks )
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

class DTStatusFlagId {

 public:

  DTStatusFlagId();
  ~DTStatusFlagId();

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

};


class DTStatusFlagData {

 public:

  DTStatusFlagData();
  ~DTStatusFlagData();

  bool noiseFlag;
  bool    feMask;
  bool   tdcMask;
  bool  trigMask;
  bool  deadFlag;
  bool  nohvFlag;

};


class DTStatusFlagCompare {
 public:
  bool operator()( const DTStatusFlagId& idl,
                   const DTStatusFlagId& idr ) const;
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
                  bool&  nohvFlag ) const
      { return get( wheelId, stationId, sectorId, slId, layerId, cellId,
                    noiseFlag, feMask, tdcMask, trigMask,
                    deadFlag, nohvFlag); };
  int cellStatus( const DTWireId& id,
                  bool& noiseFlag,
                  bool&    feMask,
                  bool&   tdcMask,
                  bool&  trigMask,
                  bool&  deadFlag,
                  bool&  nohvFlag ) const
      { return get( id,
                    noiseFlag, feMask, tdcMask, trigMask,
                    deadFlag, nohvFlag ); };
  int get( int   wheelId,
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
  int get( const DTWireId& id,
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
                     bool  nohvFlag )
      { return set( wheelId, stationId, sectorId, slId, layerId, cellId,
                    noiseFlag, feMask, tdcMask, trigMask,
                    deadFlag, nohvFlag); };
  int setCellStatus( const DTWireId& id,
                     bool noiseFlag,
                     bool    feMask,
                     bool   tdcMask,
                     bool  trigMask,
                     bool  deadFlag,
                     bool  nohvFlag )
      { return set( id,
                    noiseFlag, feMask, tdcMask, trigMask,
                    deadFlag, nohvFlag ); };

  int set( int   wheelId,
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
  int set( const DTWireId& id,
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
  typedef std::vector< std::pair<DTStatusFlagId,
                                 DTStatusFlagData> >::const_iterator
                                                      const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

  void initialize();

 private:

  DTStatusFlag(DTStatusFlag const&);
  DTStatusFlag& operator=(DTStatusFlag const&);

  std::string dataVersion;

  std::vector< std::pair<DTStatusFlagId,DTStatusFlagData> > dataList;

  edm::ConstRespectingPtr<DTBufferTree<int,int> > dBuf;

  std::string mapName() const;

};
#endif // DTStatusFlag_H

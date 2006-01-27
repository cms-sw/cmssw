#ifndef DTReadOutMapping_H
#define DTReadOutMapping_H
/** \class DTReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical drift tubes
 *
 *  $Date: 2005/11/15 13:52:00 $
 *  $Revision: 1.2 $
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
#include <vector>
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTReadOutGeometryLink {

 public:

  DTReadOutGeometryLink();
  ~DTReadOutGeometryLink();

  int     dduId;
  int     rosId;
  int     robId;
  int     tdcId;
  int channelId;
  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

};


class DTReadOutMapping {

 public:

  /** Constructor
   */
  DTReadOutMapping();
  DTReadOutMapping( const std::string& cell_map_version,
                    const std::string&  rob_map_version );

  /** Destructor
   */
  ~DTReadOutMapping();

  /** Operations
   */
  /// read and store full content
  void initSetup() const;

  /// transform identifiers
  DTWireId readOutToGeometry( int      dduId,
                              int      rosId,
                              int      robId,
                              int      tdcId,
                              int  channelId ) const;

  void readOutToGeometry( int      dduId,
                          int      rosId,
                          int      robId,
                          int      tdcId,
                          int  channelId,
                          int&   wheelId,
                          int& stationId,
                          int&  sectorId,
                          int&      slId,
                          int&   layerId,
                          int&    cellId ) const;

  void geometryToReadOut( int    wheelId,
                          int  stationId,
                          int   sectorId,
                          int       slId,
                          int    layerId,
                          int     cellId,
                          int&     dduId,
                          int&     rosId,
                          int&     robId,
                          int&     tdcId,
                          int& channelId ) const;

  /// access parent maps identifiers
  const
  std::string& mapCellTdc() const;
  std::string& mapCellTdc();
  const
  std::string& mapRobRos() const;
  std::string& mapRobRos();

  /// clear map
  void clear();

  /// insert connection
  int insertReadOutGeometryLink( int     dduId,
                                 int     rosId,
                                 int     robId,
                                 int     tdcId,
                                 int channelId,
                                 int   wheelId,
                                 int stationId,
                                 int  sectorId,
                                 int      slId,
                                 int   layerId,
                                 int    cellId );

  /// Access methods to the connections
  typedef std::vector<DTReadOutGeometryLink>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  std::string cellMapVersion;
  std::string  robMapVersion;

  std::vector<DTReadOutGeometryLink> readOutChannelDriftTubeMap;
  void getIdNumbers( int& minWheel,   int& minStation,
                     int& minSector,  int& minSL,
                     int& minLayer,   int& minCell,
                     int& minDDU,     int& minROS,
                     int& minROB,     int& minTDC,
                     int& minChannel,
                     int& maxWheel,   int& maxStation,
                     int& maxSector,  int& maxSL,
                     int& maxLayer,   int& maxCell,
                     int& maxDDU,     int& maxROS,
                     int& maxROB,     int& maxTDC,
                     int& maxChannel ) const;

};


#endif // DTReadOutMapping_H


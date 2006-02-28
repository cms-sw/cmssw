#ifndef DTReadOutMapping_H
#define DTReadOutMapping_H
/** \class DTReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical drift tubes
 *
 *  $Date: 2006/01/27 15:21:15 $
 *  $Revision: 1.3 $
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

 public:
  std::string cellMapVersion;
  std::string  robMapVersion;

  std::vector<DTReadOutGeometryLink> readOutChannelDriftTubeMap;

};


#endif // DTReadOutMapping_H


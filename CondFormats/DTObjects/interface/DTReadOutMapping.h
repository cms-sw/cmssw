#ifndef DTReadOutMapping_H
#define DTReadOutMapping_H
/** \class DTReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical drift tubes
 *
 *  $Date: 2010/01/20 18:20:08 $
 *  $Revision: 1.7 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/DTObjects/interface/DTBufferTree.h"
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


 COND_SERIALIZABLE;
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

  enum type { plain, compact };

  /** Operations
   */
  /// transform identifiers
  int readOutToGeometry( int      dduId,
                         int      rosId,
                         int      robId,
                         int      tdcId,
                         int  channelId,
                         DTWireId& wireId ) const;

  int readOutToGeometry( int      dduId,
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

  int geometryToReadOut( int    wheelId,
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
  int geometryToReadOut( const DTWireId& wireId,
                         int&     dduId,
                         int&     rosId,
                         int&     robId,
                         int&     tdcId,
                         int& channelId ) const;

  type mapType() const;

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

  /// Expand to full map
  const DTReadOutMapping* fullMap() const;

 private:

  std::string cellMapVersion;
  std::string  robMapVersion;

  std::vector<DTReadOutGeometryLink> readOutChannelDriftTubeMap;

  DTBufferTree<int,int>* mType COND_TRANSIENT;
  DTBufferTree<int,int>* rgBuf COND_TRANSIENT;
  DTBufferTree<int,int>* rgROB COND_TRANSIENT;
  DTBufferTree<int,int>* rgROS COND_TRANSIENT;
  DTBufferTree<int,int>* rgDDU COND_TRANSIENT;
  DTBufferTree<int,int>* grBuf COND_TRANSIENT;
  DTBufferTree<int,
     std::vector<int>*>* grROB COND_TRANSIENT;
  DTBufferTree<int,
     std::vector<int>*>* grROS COND_TRANSIENT;
  DTBufferTree<int,
     std::vector<int>*>* grDDU COND_TRANSIENT;

  /// read and store full content
  void cacheMap() const;
  std::string mapNameRG() const;
  std::string mapNameGR() const;


 COND_SERIALIZABLE;
};


#endif // DTReadOutMapping_H


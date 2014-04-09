#ifndef DTReadOutMapping_H
#define DTReadOutMapping_H
/** \class DTReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical drift tubes
 *       Many details related to this class are described in
 *       internal note IN 2010_033. In particular the compact
 *       format is described there.
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
#include "DataFormats/Common/interface/AtomicPtrCache.h"
#include "FWCore/Utilities/interface/ConstRespectingPtr.h"

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <string>

class DTReadOutMappingCache;
template <class Key, class Content> class DTBufferTree;

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

  DTReadOutMapping(DTReadOutMapping const&);
  DTReadOutMapping& operator=(DTReadOutMapping const&);

  edm::AtomicPtrCache<DTReadOutMappingCache> const& atomicCache() const { return atomicCache_; }
  edm::AtomicPtrCache<DTReadOutMappingCache> & atomicCache() { return atomicCache_; }

  static DTReadOutMapping* expandMap( const DTReadOutMapping& compMap );

  std::string cellMapVersion;
  std::string  robMapVersion;

  std::vector<DTReadOutGeometryLink> readOutChannelDriftTubeMap;

  edm::AtomicPtrCache<DTReadOutMappingCache> atomicCache_;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > rgBuf;
  edm::ConstRespectingPtr<DTBufferTree<int,int> > grBuf;

  /// read and store full content
  void cacheMap() const;

  std::string mapNameRG() const;
  std::string mapNameGR() const;

};
#endif // DTReadOutMapping_H

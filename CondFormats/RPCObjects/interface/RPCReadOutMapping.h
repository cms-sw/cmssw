#ifndef RPCReadOutMapping_H
#define RPCReadOutMapping_H
/** \class RPCReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical RPC strips
 *
 *  $Date: 2006/03/16 17:50 $
 *  $Revision: 1.1 $
 *  \author Marcello Maggi -- INFN Bari
 *
 */

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <vector>
#include <string>

class RPCReadOutLink {

 public:

  RPCReadOutLink();
  ~RPCReadOutLink();

  int     dccId;
  int     tbId;
  int     lboxId;
  int     mbId;
  int     lboardId;
  int channelId;
  int regionId;
  int   diskId;
  int stationId;
  int  sectorId;
  int   layerId;
  int  subsectorId;
  int    rollId;
  int stripId;
};    



class RPCReadOutMapping {

 public:

  RPCReadOutMapping();
  ~RPCReadOutMapping();

  /// read and store full content
  void initSetup() const;

  /// transform identifiers
  RPCDetId readOutToGeometry( int      dccId,
                              int      tbId,
                              int      lboxId,
                              int      mbId,
			       int lboardId,
                              int  channelId ) const;

  /// clear map
  void clear();

  /// insert connection
  int insertReadOutGeometryLink( int     dccId,
                                 int     tbId,
                                 int     lboxId,
                                 int     mbId,
				 int lboard,
                                 int channelId,
				 int regionId,
                                 int   diskId,
                                 int stationId,
                                 int  sectorId,
                                 int   layerId,
                                 int    subsectorId,
                                 int    rollId,
                                 int    stripId );

  /// Access methods to the connections
  typedef std::vector<RPCReadOutLink>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

 public:
  std::string cellMapVersion;
  std::string  robMapVersion;

  std::vector<RPCReadOutLink> readOutRPCMap;

};


#endif // RPCReadOutMapping_H


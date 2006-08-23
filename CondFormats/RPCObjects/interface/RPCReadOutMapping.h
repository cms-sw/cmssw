#ifndef RPCReadOutMapping_H
#define RPCReadOutMapping_H
/** \class RPCReadOutMapping
 *
 *  Description:
 *       Class to map read-out channels to physical RPC strips
 *
 *  $Date: 2006/03/16 17:09:57 $
 *  $Revision: 1.1 $
 *  \author Marcello Maggi -- INFN Bari
 *
 */

#include <vector>
#include <map>
#include <string>

class RPCdeteIndex;
class RPCelecIndex;
class RPCReadOutLink {

 public:

  RPCReadOutLink();
  ~RPCReadOutLink();

  int       dccId;
  int        tbId;
  int      lboxId;
  int        mbId;
  int    lboardId;
  int   channelId;
  int    regionId;
  int      diskId;
  int   stationId;
  int    sectorId;
  int     layerId;
  int subsectorId;
  int      rollId;
  int     stripId;
};    



class RPCReadOutMapping {

 public:

  RPCReadOutMapping();
  ~RPCReadOutMapping();

  /// read and store full content
  void initSetup();

  void readOutToGeometry( int        dccId,
			  int         tbId,
			  int       lboxId,
			  int         mbId,
			  int     lboardId,
			  int    channelId,
			  int&    regionId,
			  int&      diskId,
			  int&   stationId,
			  int&    sectorId,
			  int&     layerId,
			  int& subsectorId,
			  int&      rollId,
			  int&     stripId);

  /// clear map
  void clear();

  /// insert connection
  void insertReadOutGeometryLink( int       dccId,
				  int        tbId,
				  int      lboxId,
				  int        mbId,
				  int      lboard,
				  int   channelId,
				  int    regionId,
				  int      diskId,
				  int   stationId,
				  int    sectorId,
				  int     layerId,
				  int subsectorId,
				  int      rollId,
				  int     stripId );
  /// Access methods to the connections
  typedef std::vector<RPCReadOutLink>::const_iterator const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 public:
  std::vector<RPCReadOutLink> readOutRPCMap;

 private:
  std::map<RPCdeteIndex,RPCelecIndex> dtoe;
  std::map<RPCelecIndex,RPCdeteIndex> etod;


};


#endif // RPCReadOutMapping_H


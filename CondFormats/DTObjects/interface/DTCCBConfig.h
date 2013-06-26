#ifndef DTCCBConfig_H
#define DTCCBConfig_H
/** \class DTCCBConfig
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
 *
 *  $Date: 2013/05/23 15:59:44 $
 *  $Revision: 1.5 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "CondFormats/DTObjects/interface/DTBufferTree.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTCCBId {

 public:

  DTCCBId();
  ~DTCCBId();

  int   wheelId;
  int stationId;
  int  sectorId;

};


class DTConfigKey {

 public:

  DTConfigKey();
  ~DTConfigKey();

  int confType;
  int confKey;

};


class DTCCBConfig {

 public:

  /** Constructor
   */
  DTCCBConfig();
  DTCCBConfig( const std::string& version );

  /** Destructor
   */
  virtual ~DTCCBConfig();

  /** Operations
   */
  /// get content
  std::vector<DTConfigKey> fullKey() const;
  int stamp() const;
  int configKey( int   wheelId,
                 int stationId,
                 int  sectorId,
                 std::vector<int>& confKey ) const;
  int configKey( const DTChamberId& id,
                 std::vector<int>& confKey ) const;
  typedef std::vector< std::pair< DTCCBId,std::vector<int> > > ccb_config_map;
  typedef ccb_config_map::const_iterator ccb_config_iterator;
  ccb_config_map configKeyMap() const;

  /// access version
  const
  std::string& version() const;
  std::string& version();

  /// reset content
  void clear();

  void setFullKey( const std::vector<DTConfigKey>& );
  void setStamp( int s );

  int setConfigKey( int   wheelId,
                    int stationId,
                    int  sectorId,
                    const std::vector<int>& confKey );
  int setConfigKey( const DTChamberId& id,
                    const std::vector<int>& confKey );

  int appendConfigKey( int   wheelId,
                       int stationId,
                       int  sectorId,
                       const std::vector<int>& confKey );
  int appendConfigKey( const DTChamberId& id,
                       const std::vector<int>& confKey );

  /// Access methods to data
  typedef std::vector< std::pair<DTCCBId,int> >::const_iterator
                                                 const_iterator;
  const_iterator begin() const;
  const_iterator end() const;

 private:

  int timeStamp;
  std::string dataVersion;
  std::vector<DTConfigKey> fullConfigKey;
  std::vector< std::pair<DTCCBId,int> > dataList;

  DTBufferTree< int,std::vector<int>* >* dBuf;

  /// read and store full content
  void cacheMap() const;
  void resetMap() const;
//  std::string mapName() const;

};


#endif // DTCCBConfig_H


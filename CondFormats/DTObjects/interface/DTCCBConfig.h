#ifndef DTCCBConfig_H
#define DTCCBConfig_H
/** \class DTCCBConfig
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
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
#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "FWCore/Utilities/interface/ConstRespectingPtr.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <utility>

class DTBufferTreeUniquePtr;

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


 COND_SERIALIZABLE;
};


class DTConfigKey {

 public:

  DTConfigKey();
  ~DTConfigKey();

  int confType;
  int confKey;


 COND_SERIALIZABLE;
};


class DTCCBConfig {

 public:

  DTCCBConfig();
  DTCCBConfig( const std::string& version );

  virtual ~DTCCBConfig();

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

  void initialize();

 private:

  DTCCBConfig(DTCCBConfig const&) = delete;
  DTCCBConfig& operator=(DTCCBConfig const&) = delete;

  int timeStamp;
  std::string dataVersion;
  std::vector<DTConfigKey> fullConfigKey;
  std::vector< std::pair<DTCCBId,int> > dataList;

  edm::ConstRespectingPtr<DTBufferTreeUniquePtr> dBuf COND_TRANSIENT;

 COND_SERIALIZABLE;
};
#endif // DTCCBConfig_H

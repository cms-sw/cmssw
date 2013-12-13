#ifndef DTKeyedConfigHandler_H
#define DTKeyedConfigHandler_H
/** \class DTKeyedConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/05/14 11:43:08 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "CondCore/PopCon/interface/PopConSourceHandler.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include <string>

namespace cond {
  namespace persistency {
    class KeyList;
  }
}

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTKeyedConfigHandler: public popcon::PopConSourceHandler<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTKeyedConfigHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTKeyedConfigHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

  static void setList( cond::persistency::KeyList* list );

 private:

  bool copyData;
  int minBrickId;
  int maxBrickId;
  int minRunId;
  int maxRunId;

  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string brickContainer;
  DTCCBConfig* ccbConfig;
  
  cond::persistency::ConnectionPool connection;
  cond::persistency::Session isession;
  void chkConfigList();
  static bool sameConfigList( const std::vector<DTConfigKey>& cfgl,
                              const std::vector<DTConfigKey>& cfgr );

  static cond::persistency::KeyList* keyList;

};


#endif // DTKeyedConfigHandler_H







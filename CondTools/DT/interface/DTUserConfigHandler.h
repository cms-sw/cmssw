#ifndef DTUserConfigHandler_H
#define DTUserConfigHandler_H
/** \class DTUserConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2009/09/25 12:03:21 $
 *  $Revision: 1.4 $
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
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include <string>

namespace coral {
  class ISessionProxy;
}

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTUserConfigHandler: public popcon::PopConSourceHandler<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTUserConfigHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTUserConfigHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  int         dataRun;
  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string offlineAuthentication;
  std::string offlineConnect;
  std::string offlineCatalog;
  std::string listToken;
  std::vector<DTConfigKey> userConf;
  DTCCBConfig* ccbConfig;

  coral::ISessionProxy* isession;
  void chkConfigList( const std::map<int,bool>& userBricks );
  bool userDiscardedKey( int key );
  static bool sameConfigList( const std::vector<DTConfigKey>& cfgl,
                              const std::vector<DTConfigKey>& cfgr );

};


#endif // DTUserConfigHandler_H







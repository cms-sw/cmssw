#ifndef DTUserKeyedConfigHandler_H
#define DTUserKeyedConfigHandler_H
/** \class DTUserKeyedConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/03/18 16:06:38 $
 *  $Revision: 1.1.2.1 $
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

namespace cond {
  class KeyList;
}

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTUserKeyedConfigHandler: public popcon::PopConSourceHandler<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTUserKeyedConfigHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTUserKeyedConfigHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

  static void setList( cond::KeyList* list );

 private:

  int         dataRun;
  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string brickContainer;
  std::vector<DTConfigKey> userConf;
  DTCCBConfig* ccbConfig;

  coral::ISessionProxy* isession;
  void chkConfigList( const std::map<int,bool>& userBricks );
  bool userDiscardedKey( int key );
  static bool sameConfigList( const std::vector<DTConfigKey>& cfgl,
                              const std::vector<DTConfigKey>& cfgr );

  static cond::KeyList* keyList;

};


#endif // DTUserKeyedConfigHandler_H







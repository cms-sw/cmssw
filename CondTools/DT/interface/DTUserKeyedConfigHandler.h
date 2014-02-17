#ifndef DTUserKeyedConfigHandler_H
#define DTUserKeyedConfigHandler_H
/** \class DTUserKeyedConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/07/21 16:06:53 $
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
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondFormats/DTObjects/interface/DTCCBConfig.h"
#include <string>

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
  bool writeKeys;
  bool writeData;
  DTCCBConfig* ccbConfig;
  
  cond::DbConnection connection;
  cond::DbSession isession;
  void chkConfigList( const std::map<int,bool>& userBricks );
  bool userDiscardedKey( int key );
  static bool sameConfigList( const std::vector<DTConfigKey>& cfgl,
                              const std::vector<DTConfigKey>& cfgr );

  static cond::KeyList* keyList;

};


#endif // DTUserKeyedConfigHandler_H







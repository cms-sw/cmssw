#ifndef DTUserKeyedConfigHandler_H
#define DTUserKeyedConfigHandler_H
/** \class DTUserKeyedConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2010/07/21 17:13:43 $
 *  $Revision: 1.3 $
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
#include <memory>
#include <string>

namespace coral {
  class ISessionProxy;
}

namespace cond {
  namespace persistency {
    class KeyList;
  }
}  // namespace cond

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTUserKeyedConfigHandler : public popcon::PopConSourceHandler<DTCCBConfig> {
public:
  /** Constructor
   */
  DTUserKeyedConfigHandler(const edm::ParameterSet& ps);

  /** Destructor
   */
  ~DTUserKeyedConfigHandler() override;

  /** Operations
   */
  ///
  void getNewObjects() override;
  std::string id() const override;

  void setList(const cond::persistency::KeyList* list);

private:
  int dataRun;
  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  int onlineAuthSys;
  std::string brickContainer;
  std::vector<DTConfigKey> userConf;
  bool writeKeys;
  bool writeData;
  DTCCBConfig* ccbConfig;

  cond::persistency::ConnectionPool connection;
  std::shared_ptr<coral::ISessionProxy> isession;
  void chkConfigList(const std::map<int, bool>& userBricks);
  bool userDiscardedKey(int key);
  static bool sameConfigList(const std::vector<DTConfigKey>& cfgl, const std::vector<DTConfigKey>& cfgr);

  const cond::persistency::KeyList* keyList = nullptr;
};

#endif  // DTUserKeyedConfigHandler_H

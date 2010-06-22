#ifndef DTCCBConfigHandler_H
#define DTCCBConfigHandler_H
/** \class DTCCBConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2008/02/15 18:15:02 $
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

class DTCCBConfigHandler: public popcon::PopConSourceHandler<DTCCBConfig> {

 public:

  /** Constructor
   */
  DTCCBConfigHandler( const edm::ParameterSet& ps );

  /** Destructor
   */
  virtual ~DTCCBConfigHandler();

  /** Operations
   */
  /// 
  void getNewObjects();
  std::string id() const;

 private:

  std::string dataTag;
  std::string onlineConnect;
  std::string onlineAuthentication;
  std::string offlineAuthentication;
  std::string offlineConnect;
  std::string offlineCatalog;
  std::string listToken;
  DTCCBConfig* ccbConfig;

  coral::ISessionProxy* isession;
  void chkConfigList();
  static bool sameConfigList( const std::vector<DTConfigKey>& cfgl,
                              const std::vector<DTConfigKey>& cfgr );

};


#endif // DTCCBConfigHandler_H







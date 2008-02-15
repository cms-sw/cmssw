#ifndef DTCCBConfigHandler_H
#define DTCCBConfigHandler_H
/** \class DTCCBConfigHandler
 *
 *  Description: 
 *
 *
 *  $Date: 2007/12/07 15:12:13 $
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

};


#endif // DTCCBConfigHandler_H







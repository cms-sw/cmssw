#ifndef DTDBSession_H
#define DTDBSession_H
/** \class DTDBSession
 *
 *  Description: 
 *
 *
 *  $Date: 2007/11/24 12:29:51 $
 *  $Revision: 1.1.4.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/Connection.h"
//#include "CondCore/DBCommon/interface/PoolStorageManager.h"
//#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBOutputService/interface/serviceCallbackRecord.h"
#include <string>
#include <map>
namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}
namespace cond{
  class PoolTransaction;
  class Connection;
}

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDBSession {

 public:

  /** Constructor
   */
  DTDBSession( const std::string& dbFile,
               const std::string& dbCatalog,
               const std::string& auth_path,
               bool siteLocalConfig = false );

  /** Destructor
   */
  virtual ~DTDBSession();

  /** Operations
   */
  /// get storage manager
  cond::PoolTransaction* poolDB() const;
//  cond::PoolStorageManager* poolDB() const;

  /// start transaction
  void connect( bool readOnly );
  /// end   transaction
  void disconnect();

 private:

  cond::DBSession* m_session;
  cond::Connection* m_connection;
  cond::PoolTransaction* m_pooldb;
//  cond::PoolStorageManager* m_pooldb;

};


#endif // DTDBSession_H







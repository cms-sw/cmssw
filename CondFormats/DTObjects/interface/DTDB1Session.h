#ifndef DTDB1Session_H
#define DTDB1Session_H
/** \class DTDB1Session
 *
 *  Description: 
 *
 *  TEMPORARY TOOL TO HANDLE CONFIGURATIONS
 *  TO BE REMOVED IN FUTURE RELEASES
 *
 *
 *  $Date: 2009/05/28 17:18:56 $
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
//#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
//#include "CondCore/DBCommon/interface/DBSession.h"
//#include "CondCore/IOVService/interface/IOVService.h"
//#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
//#include "CondCore/DBCommon/interface/Connection.h"
////#include "CondCore/DBCommon/interface/PoolStorageManager.h"
////#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
//#include "CondCore/DBCommon/interface/TypedRef.h"
//#include "CondCore/DBCommon/interface/Time.h"
//#include "CondCore/MetaDataService/interface/MetaData.h"
//#include "CondCore/DBOutputService/interface/serviceCallbackRecord.h"
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
  class DBSession;
}

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTDB1Session {

 public:

  /** Constructor
   */
  DTDB1Session( const std::string& dbFile,
               const std::string& dbCatalog,
               const std::string& auth_path,
               bool siteLocalConfig = false );

  /** Destructor
   */
  virtual ~DTDB1Session();

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


#endif // DTDB1Session_H







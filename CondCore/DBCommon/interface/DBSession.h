#ifndef COND_DBCommon_DBSession_h
#define COND_DBCommon_DBSession_h
//
// Package:    CondCore/DBCommon
// Class:      DBSession
//
/**\class DBSession DBSession.h CondCore/DBCommon/interface/DBSession.h
 Description: Class to prepare database connection setup
*/
//
// Author:      Zhen Xie
//
#include <string>
//#include "CoralKernel/Context.h"
//#include "CoralKernel/IHandle.h"

#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/CoralServiceManager.h"


namespace coral{
  class IConnectionService;
  class IRelationalService;
  class IAuthenticationService;
  class IMonitoringReporter;
  class IWebCacheControl;
}
namespace pool{
  class IBlobStreamingService;
}
namespace cond{

  // session configurartion, later more code, now just one set of defaults
  // move to SessionConfiguration?
  enum ConfDefaults { coralDefaults, cmsDefaults, prodDefaults, toolDefaults, webDefaults}; 

  /*
  **/
  class DBSession{
  public:
    DBSession();
    DBSession(ConfDefaults confDef = /*cmsDefaults */);
    ~DBSession();
    void open();
    //void close();

    void config(ConfDefaults confDef);

    coral::IConnectionService& connectionService();
    coral::IRelationalService& relationalService();
    coral::IAuthenticationService& authenticationService() ;
    const coral::IMonitoringReporter& monitoringReporter() const;
    coral::IWebCacheControl& webCacheControl();
    pool::IBlobStreamingService& blobStreamingService();
    cond::SessionConfiguration& configuration();
  private:
    //    coral::IHandle<coral::Context> m_context;
    SessionConfiguration m_sessionConfig;
    CoralServiceManager m_pluginmanager;
  };
}//ns cond
#endif
// DBSESSION_H

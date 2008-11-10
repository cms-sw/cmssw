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
// $Id$
//
#include <string>
#include "CoralKernel/Context.h"
#include "CoralKernel/IHandle.h"
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
  class SessionConfiguration;
  /*
  **/
  class DBSession{
  public:
    DBSession();
    ~DBSession();
    void open();
    //void close();
    coral::IConnectionService& connectionService();
    coral::IRelationalService& relationalService();
    coral::IAuthenticationService& authenticationService() ;
    const coral::IMonitoringReporter& monitoringReporter() ;
    coral::IWebCacheControl& webCacheControl();
    pool::IBlobStreamingService& blobStreamingService();
    cond::SessionConfiguration& configuration();
  private:
    //    coral::IHandle<coral::Context> m_context;
    SessionConfiguration* m_sessionConfig;
  };
}//ns cond
#endif
// DBSESSION_H

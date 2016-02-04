#ifndef CondCore_IOVSchemaUtility_h
#define CondCore_IOVSchemaUtility_h

#include "CondCore/DBCommon/interface/DbSession.h"
namespace cond{
  class IOVSchemaUtility{
  public:
    explicit IOVSchemaUtility(DbSession& session);
    IOVSchemaUtility(DbSession& session, std::ostream& log);
    ~IOVSchemaUtility();
    /// create iov tables if not existing
    bool createIOVContainerIfNecessary();

    /// drop iov tables if existing
    bool dropIOVContainer();

    /// create a payload container
    void createPayloadContainer( const std::string& payloadName, const std::string& payloadTypeName );

    /// drop iov tables if existing
    void dropPayloadContainer( const std::string& payloadName );

    /// drop all
    void dropAll();
  private:
    cond::DbSession& m_session;
    std::ostream* m_log;
  };
}//ns cond
#endif

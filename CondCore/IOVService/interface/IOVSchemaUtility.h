#ifndef CondCore_IOVSchemaUtility_h
#define CondCore_IOVSchemaUtility_h

#include "CondCore/DBCommon/interface/DbSession.h"
namespace cond{
  class IOVSchemaUtility{
  public:
    IOVSchemaUtility(DbSession& pooldb);
    ~IOVSchemaUtility();
    /// create iov tables if not existing
    void create();
    /// drop iov tables if existing
    void drop();
    /// truncate iov tables if existing
    void truncate();
  private:
    cond::DbSession m_pooldb;
  };
}//ns cond
#endif

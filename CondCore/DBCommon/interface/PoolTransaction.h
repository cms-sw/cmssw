#ifndef CondCore_DBCommon_PoolTransaction_H
#define CondCore_DBCommon_PoolTransaction_H
#include "CondCore/DBCommon/interface/ITransaction.h"
namespace pool{
  class IDataSvc;
}
namespace cond{
  class PoolConnectionProxy;
  /**
     PoolTransaction 
  */
  class PoolTransaction : public ITransaction{
  public:
    explicit PoolTransaction(cond::PoolConnectionProxy* parentConnection);
    ~PoolTransaction();
    void start();
    void commit();
    void rollback();
    virtual bool isReadOnly() const;
    virtual IConnectionProxy& parentConnection();
    void resetPoolDataSvc(pool::IDataSvc* datasvc) const;
    pool::IDataSvc& poolDataSvc();
  private:
    cond::PoolConnectionProxy* m_parentConnection;
    mutable pool::IDataSvc* m_datasvc;
    mutable bool m_isReadOnly;
  };
}
#endif

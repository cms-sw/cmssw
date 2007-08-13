#ifndef CondCore_DBCommon_CoralTransaction_H
#define CondCore_DBCommon_CoralTransaction_H
#include "CondCore/DBCommon/interface/ITransaction.h"
#include <vector>
namespace coral{
  class ISessionProxy;
  class ISchema;
}
namespace cond{
  class CoralConnectionProxy;
  /**
     CoralTransaction transaction is child of connection and subject for observers
  */
  class CoralTransaction : public ITransaction{
  public:
    CoralTransaction(CoralConnectionProxy* parentConnection);
    ~CoralTransaction();
    virtual void start();
    virtual void commit();
    virtual void rollback();
    virtual bool isReadOnly() const;
    virtual IConnectionProxy& parentConnection();
    coral::ISchema& nominalSchema();
    coral::ISessionProxy& coralSessionProxy();
    void resetCoralHandle(coral::ISessionProxy* coralHandle) const;
  private:
    CoralConnectionProxy* m_parentConnection;
    /// coral sessionproxy handle. parent connection has the ownership
    mutable coral::ISessionProxy* m_coralHandle;
    mutable bool m_isReadOnly;
  };
}
#endif

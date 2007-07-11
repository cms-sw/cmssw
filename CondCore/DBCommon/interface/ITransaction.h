#ifndef CondCore_DBCommon_ITransaction_H
#define CondCore_DBCommon_ITransaction_H
#include <vector>
namespace cond{
  class IConnectionProxy;
  class ITransactionObserver;
  /**
     abstract Transaction interface
  */
  class ITransaction{
  public:
    ITransaction(){
      m_observers.reserve(10);
    }
    virtual ~ITransaction(){}
    virtual void start() = 0;
    virtual void commit() = 0;
    virtual void rollback() = 0;
    virtual bool isReadOnly() const = 0;
    virtual IConnectionProxy& parentConnection() = 0;
  protected:
    void attach( ITransactionObserver* );
    void NotifyStartOfTransaction( ) ;
    void NotifyEndOfTransaction() ;
  private:
    std::vector< cond::ITransactionObserver* > m_observers;
  };
}
#endif

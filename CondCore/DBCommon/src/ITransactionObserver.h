#ifndef CondCore_DBCommon_ITransactionObserver_H
#define CondCore_DBCommon_ITransactionObserver_H
namespace cond{
  class ITransaction;
  /* defines transaction observer interface
  **/
  class ITransactionObserver{
  public:
    ITransactionObserver(){}
    virtual ~ITransactionObserver(){}
    virtual void reactOnStartOfTransaction( const ITransaction* ) = 0;
    virtual void reactOnEndOfTransaction( const ITransaction* ) = 0;    
  };
}
#endif

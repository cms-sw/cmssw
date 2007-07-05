#ifndef CondCore_DBCommon_IConnectionProxy_H
#define CondCore_DBCommon_IConnectionProxy_H
#include <string>
namespace cond{
  class ITransaction;
  /* abstract interface for real connection
  **/
  class IConnectionProxy{
  public:
    IConnectionProxy(){}
    virtual ~IConnectionProxy(){}
    virtual ITransaction&  transaction() = 0;
    virtual bool isReadOnly() const = 0;
    virtual unsigned int connectionTimeOut() const = 0;
    virtual std::string connectStr() const = 0;
  };
}
#endif

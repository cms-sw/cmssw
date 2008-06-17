#ifndef  PayLoadInspector_H
#define  PayLoadInspector_H
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include <string>

namespace cond {


  template<typename T>
  class PayLoadInspector {
  public:
    typedef T Class;
    
    PayLoadInspector() {}
    PayLoadInspector(const cond::IOVElement & elem) : 
      object(*elem.db(),elem.payloadToken()){}

    std::string print() const;

    std::string summary() const;

  private:
    cond::TypedRef<Class> object;    

  };

}

#endif //   PayLoadInspector_H

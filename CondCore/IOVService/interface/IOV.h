#ifndef IOV_H
#define IOV_H

#include<map>
#include <string>
namespace cond{
  class IOV {
  public:
    IOV(){}
    virtual ~IOV(){}
    std::map<unsigned long,std::string> iov;
  };
}
#endif

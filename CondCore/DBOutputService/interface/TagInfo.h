#ifndef CondCore_DBOutputService_TagInfo_H
#define CondCore_DBOutputService_TagInfo_H
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class TagInfo{
  public:
    TagInfo(){}
    std::string name;
    std::string token;
    cond::ValidityInterval lastInterval;
    size_t size;
  };
}
#endif

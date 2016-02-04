#ifndef CondCore_DBCommon_TagInfo_H
#define CondCore_DBCommon_TagInfo_H
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class TagInfo{
  public:
    TagInfo(): lastInterval(0,0), size(0){}
    std::string name;
    std::string token;
    cond::ValidityInterval lastInterval;
    std::string lastPayloadToken;
    size_t size;
  };
}
#endif

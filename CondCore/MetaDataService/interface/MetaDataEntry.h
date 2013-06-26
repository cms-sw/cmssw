#ifndef CondCore_MetaDataService_MetaDataEntry_H
#define CondCore_MetaDataService_MetaDataEntry_H
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class MetaDataEntry{
  public:
    std::string tagname;
    std::string iovtoken;
    cond::TimeType timetype;
  };
}
#endif

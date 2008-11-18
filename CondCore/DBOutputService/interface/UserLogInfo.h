#ifndef CondCore_DBOutputService_USERLOGINFO_H
#define CondCore_DBOutputService_USERLOGINFO_H
#include <string>
namespace cond{
  namespace service{
    class UserLogInfo{
    public:
      std::string provenance;
      std::string usertext;
    };
    class NullUserLogInfo : public UserLogInfo{
    };
  }
}
#endif

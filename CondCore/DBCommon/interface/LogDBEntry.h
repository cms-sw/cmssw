#ifndef CondCore_DBCommon_LogDBEntry_H
#define CondCore_DBCommon_LogDBEntry_H
#include <string>
namespace cond{

  class UserLogInfo{
    public:
      std::string provenance;
      std::string usertext;
  };

  class NullUserLogInfo : public UserLogInfo{
  };

  class LogDBEntry{
  public:
    unsigned long long logId;
    std::string destinationDB;   
    std::string provenance;
    std::string usertext;
    std::string iovtag;
    std::string iovtimetype;
    unsigned int payloadIdx;
    unsigned long long lastSince;
    std::string payloadClass;
    std::string payloadToken;
    std::string exectime;
    std::string execmessage;
  };
}
#endif

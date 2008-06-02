#ifndef CondCore_DBCommon_LogDBEntry_H
#define CondCore_DBCommon_LogDBEntry_H
#include <string>
namespace cond{
  class LogDBEntry{
  public:
    unsigned long long logId;
    std::string destinationDB;   
    std::string provenance;
    std::string usertext;
    std::string iovtag;
    std::string iovtimetype;
    unsigned int payloadIdx;
    std::string payloadName;
    std::string payloadToken;
    std::string payloadContainer;
    std::string exectime;
    std::string execmessage;
  };
}
#endif

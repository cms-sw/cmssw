#ifndef DBCommon_FipProtocolParser_H
#define DBCommon_FipProtocolParser_H
#include <string>
namespace cond{
  class FipProtocolParser{
  public:
    FipProtocolParser();
    ~FipProtocolParser();
    std::string getRealConnect(const std::string& fipConnect) const;
  };
}//ns cond
#endif


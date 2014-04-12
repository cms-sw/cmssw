#ifndef CondCore_DBCommon_LogDBNames_h
#define CondCore_DBCommon_LogDBNames_h
#include <string>
namespace cond{
  class LogDBNames{
  public:
    static std::string SequenceTableName();
    static std::string LogTableName();
  };
}//ns cond
#endif

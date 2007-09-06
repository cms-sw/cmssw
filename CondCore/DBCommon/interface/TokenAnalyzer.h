#ifndef DBCommon_TokenAnalyzer_h
#define DBCommon_TokenAnalyzer_h
#include <string>
namespace cond{
  class TokenAnalyzer{
  public:
    TokenAnalyzer(){}
    ~TokenAnalyzer(){}
    std::string getFID(const std::string& strToken) const;
  };
}//ns cond
#endif

#ifndef CondCore_DBCommon_ConnectStringParser_h
#define CondCore_DBCommon_ConnectStringParser_h
#include <string>
#include <vector>
namespace cond{
  class ConnectStringParser{
  public:
    explicit ConnectStringParser( const std::string& inputStr );
    ~ConnectStringParser();
    bool isLogical() const;
    std::string result() const;
    void reset( const std::string& inputStr );
  private:
    void parseLogical();
    void parsePhysical();
    std::vector< std::string > m_result;
    bool m_isLogical;
    std::string m_inputStr;
  };
}//ns cond
#endif

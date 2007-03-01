#ifndef CondTools_Utilities_CSVHeaderLineParser_h
#define CondTools_Utilities_CSVHeaderLineParser_h
#include <string>
#include <vector>
class CSVHeaderLineParser{
 public:
  CSVHeaderLineParser(){}
  ~CSVHeaderLineParser(){}
  bool parse( const std::string& inputLine);
  std::vector<std::string> result() const;
 private:
  std::vector<std::string> m_result;
};
#endif

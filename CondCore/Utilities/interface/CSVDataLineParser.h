#ifndef CondTools_Utilities_CSVDataLineParser_h
#define CondTools_Utilities_CSVDataLineParser_h
#include <string>
#include <vector>
#include <boost/any.hpp>
class CSVDataLineParser{
 public:
  CSVDataLineParser(){}
  ~CSVDataLineParser(){}
  bool parse( const std::string& inputLine);
  std::vector<boost::any> result() const;
 private:
  std::vector<boost::any> m_result;
};
#endif

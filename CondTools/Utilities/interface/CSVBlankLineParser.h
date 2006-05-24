#ifndef CondTools_Utilities_CSVBlankLineParser_h
#define CondTools_Utilities_CSVBlankLineParser_h
#include <string>
class CSVBlankLineParser{
 public:
  CSVBlankLineParser(){}
  ~CSVBlankLineParser(){}
  bool isBlank( const std::string& inputLine);
};
#endif

#ifndef CondFormats_strKeyMap_h
#define CondFormats_strKeyMap_h
#include <map>
class Algob{
 public:
  Algob(){}
  int b;
};
class strKeyMap{
 public:
  strKeyMap(){}
 private:
  std::map<std::string, Algob> m_content;
};
#endif 

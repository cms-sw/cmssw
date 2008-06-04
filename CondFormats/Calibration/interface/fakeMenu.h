#ifndef CondFormats_fakeMenu_h
#define CondFormats_fakeMenu_h
#include <map>
class Algo{
 public:
  Algo(){}
  int a;
};
class AlgoMap : public std::map<std::string, Algo>{
 public:
  AlgoMap(){}
};
class fakeMenu{
 public:
  // constructor
  fakeMenu(){}
  virtual ~fakeMenu(){}
 private:
  AlgoMap m_algorithmMap;
};
#endif 

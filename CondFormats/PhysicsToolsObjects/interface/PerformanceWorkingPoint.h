#ifndef PerformanceWorkingPoint_h
#define PerformanceWorkingPoint_h


#include "string"

class PerformanceWorkingPoint {
 public:
  PerformanceWorkingPoint(){}
  PerformanceWorkingPoint(float c, std::string s) : cut_(c), dname_ (s) {}
  float cut()const {return cut_;}
  std::string discriminantName()const {return dname_;}
  
 private: 
  float cut_;
  std::string dname_;
};


#endif

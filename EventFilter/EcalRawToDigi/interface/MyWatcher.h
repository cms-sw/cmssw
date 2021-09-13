#ifndef MyWATCHER_H
#define MyWATCHER_H

#include "TStopwatch.h"
#include <iostream>
#include <sstream>

#ifdef EDM_ML_DEBUG
class MyWatcher : public TStopwatch {
 public:
  MyWatcher(const std::string n=""):name(n),total(0) {}
  ~MyWatcher(){}
  
  std::string start(bool r=true){Start(r); return " [Start]";}
  std::string continu(){Continue(); return " [Continue]";}
  std::string reset(){Reset(); return " [Reset]";}
  std::string stop() {Stop(); return " [Stop]";}
  std::string lap() {
    std::stringstream o;
    double r=RealTime();
    total+=r;
    o<<"\n   "<<r<<" total:"<<total<<" ["<<name<<"]";
    Start();
    return o.str();}
  std::string name;
  double total;
};
#else
class MyWatcher {
 public:
  MyWatcher(const std::string) {}
  ~MyWatcher(){}

  std::string start(bool r=true){return name;}
  std::string continu(){return name;}
  std::string reset(){return name;}
  std::string stop(){return name;}
   std::string lap() {return name;}
std::string name;
};
#endif

#endif

#ifndef AlarmElement_h
#define AlarmElement_h


#include "DQMServices/Core/interface/MonitorElementBaseT.h"

class AlarmElement : public MonitorElementString
{

 public:
  
  AlarmElement(std::string *val,char * name = 0);
};
#endif










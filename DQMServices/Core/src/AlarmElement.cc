#include "DQMServices/Core/interface/AlarmElement.h"

using namespace std;

AlarmElement::AlarmElement(string *val,char * name) : 
  MonitorElementString(val,name) 
{
  setUrgent();
}

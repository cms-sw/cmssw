#ifndef DQM_SiStripCommon_ExtractTObject_H
#define DQM_SiStripCommon_ExtractTObject_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include <string>

/** */
template <class T> 
class ExtractTObject {
 public:
  static T* extract( MonitorElement* me ) {
    return me ? dynamic_cast<T*>(me->getRootObject()) : 0;
  }
};

#endif // DQM_SiStripCommon_ExtractTObject_H

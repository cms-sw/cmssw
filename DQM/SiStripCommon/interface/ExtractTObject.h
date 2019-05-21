#ifndef DQM_SiStripCommon_ExtractTObject_H
#define DQM_SiStripCommon_ExtractTObject_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include <string>

class MonitorElement;

/** */
template <class T>
class ExtractTObject {
public:
  static T* extract(MonitorElement* me);
};

#endif  // DQM_SiStripCommon_ExtractTObject_H

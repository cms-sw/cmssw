#ifndef DQM_SiStripCommon_ExtractTObject_H
#define DQM_SiStripCommon_ExtractTObject_H

class MonitorElement;
class TNamed;

/** */
template <class T> 
class ExtractTObject {
  
 public:
  
  static T* extract( MonitorElement* me );
  
  static T* extract( TNamed* tnamed );
  
};

#endif // DQM_SiStripCommon_ExtractTObject_H


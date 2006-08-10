#ifndef DQM_SiStripCommon_ExtractTObject_H
#define DQM_SiStripCommon_ExtractTObject_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "TNamed.h"
#include <string>

/** */
template <class T> 
class ExtractTObject {
  
 public:
  
  static T* extract( MonitorElement* me ) {
    if ( me ) {
      MonitorElementT<TNamed>* tnamed = dynamic_cast< MonitorElementT<TNamed>* >( me );
      if ( tnamed ) {
	T* histo = ExtractTObject::extract( tnamed->operator->() );
	if ( histo ) { 
	  return histo; 
	} else { return 0; }
      } else { return 0; }
    } else { return 0; }
  }
  
  static T* extract( TNamed* tnamed ) {
    if ( tnamed ) {
      T* histo = dynamic_cast<T*>( tnamed );
      if ( histo ) { 
	return histo; 
      } else { return 0; }
    } else { return 0; }
  }
  
};

#endif // DQM_SiStripCommon_ExtractTObject_H


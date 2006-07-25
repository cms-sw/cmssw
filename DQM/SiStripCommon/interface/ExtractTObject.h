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
  
  T* extract( MonitorElement* me ) { 
    if ( me ) {
      MonitorElementT<TNamed>* tnamed = dynamic_cast< MonitorElementT<TNamed>* >( me );
      if ( tnamed ) {
	T* histo = dynamic_cast<T*>( tnamed->operator->() );
	if ( histo ) { return histo; }
	else { return 0; }
      } else { return 0; }
    } else { return 0; }
  }
  
  T* insert( MonitorElement* me, T* new_histo, std::string new_name = "" ) { 
    if ( me ) {
      MonitorElementT<TNamed>* tnamed = dynamic_cast< MonitorElementT<TNamed>* >( me );
      if ( tnamed ) {
	T* old_histo = dynamic_cast<T*>( tnamed->operator->() );
	if ( old_histo ) { 
	  tnamed->insert( new_histo, new_name );
	  return tnamed->operator->(); 
	}
	else { return 0; }
      } else { return 0; }
    } else { return 0; }
  }
  
};

#endif // DQM_SiStripCommon_ExtractTObject_H


#ifndef DQM_SiStripCommon_ExtractTObject_H
#define DQM_SiStripCommon_ExtractTObject_H

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "TNamed.h"

/** */
template <class T> 
class ExtractTObject {
  
 public:
  
  T* operator() ( MonitorElement* me ) { 
    MonitorElementT<TNamed>* tnamed = dynamic_cast< MonitorElementT<TNamed>* >( me );
    if ( tnamed ) {
      T* histo = dynamic_cast<T*>( tnamed->operator->() );
      if( histo ) { return histo; }
      else { return 0; }
    } else { return 0; }
  }
  
};

#endif // DQM_SiStripCommon_ExtractTObject_H

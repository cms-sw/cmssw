#ifndef EBMUtilsClient_H
#define EBMUtilsClient_H

/*
 * \file EBMUtilsClient.h
 *
 * $Date: $
 * $Revision: $
 * \author B. Gobbo
 *
*/

#include "DQMServices/Core/interface/MonitorElement.h"
#include "TROOT.h"

/// getHisto
template <class T> static T* getHisto( const MonitorElement* me, const T* h ) {
  T* ret = 0; 
  if( me ) {
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if( ob ) { ret = dynamic_cast<T*>( ob->operator->()); }
  }
  return ret;
}

#endif // EBMUtilsClient_h

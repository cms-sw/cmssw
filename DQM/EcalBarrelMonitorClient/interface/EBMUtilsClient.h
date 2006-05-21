// $Id: EBMUtilsClient.h,v 1.4 2006/05/21 20:42:07 dellaric Exp $

/*!
  \file EBMUtilsClient.h
  \brief Ecal Barrel Monitor Utils for Client
  \author B. Gobbo 
  \version $Revision: 1.4 $
  \date $Date: 2006/05/21 20:42:07 $
*/

#ifndef EBMUtilsClient_H
#define EBMUtilsClient_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "TROOT.h"

class EBMUtilsClient {

 public:

  //! getHisto
  template<class T> static T* getHisto( const MonitorElement* me, bool clone = false, T* ret = 0 ) {
    if( me ) {
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
      if( ob ) { 
        if( clone ) {
          if( ret ) delete ret;
          ret = 0;
          std::string s = "ME " + me->getName();
          ret = dynamic_cast<T*>((ob->operator->())->Clone(s.c_str())); 
        } else {
          ret = dynamic_cast<T*>( ob->operator->()); 
        }
      }
    }
    return ret;
  }

 protected:
  EBMUtilsClient() {}

};

#endif

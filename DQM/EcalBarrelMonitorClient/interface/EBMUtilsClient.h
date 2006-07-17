// $Id: EBMUtilsClient.h,v 1.14 2006/07/03 07:16:40 dellaric Exp $

/*!
  \file EBMUtilsClient.h
  \brief Ecal Barrel Monitor Utils for Client
  \author B. Gobbo 
  \version $Revision: 1.14 $
  \date $Date: 2006/07/03 07:16:40 $
*/

#ifndef EBMUtilsClient_H
#define EBMUtilsClient_H

#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TROOT.h"

/*! \class EBMUtilsClient
    \brief Utilities for Ecal Barrel Monitor Client 
 */

class EBMUtilsClient {

 public:

  /*! \fn template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0 )
      \brief Returns the histogram contained by the Monitor Element
      \param me Monitor Element.
      \param clone (boolean) if true clone the histogram. 
      \param ret in case of clonation delete the histogram first.
   */
  template<class T> static T getHisto( const MonitorElement* me, bool clone = false, T ret = 0 ) {
    if( me ) {
      // cout << "Found '" << me->getName() <<"'"; 
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
      if( ob ) { 
        if( clone ) {
          if( ret ) {
            delete ret;
          }
          std::string s = "ME " + me->getName();
          ret = dynamic_cast<T>((ob->operator->())->Clone(s.c_str())); 
          if( ret ) {
            ret->SetDirectory(0);
          }
        } else {
          ret = dynamic_cast<T>(ob->operator->()); 
        }
      } else {
	ret = 0;
      }
    } else {
      if( !clone ) {
        ret = 0;
      }
    }
    return ret;
  }

  /*! \fn static void resetHisto( const MonitorElement* me ) {
      \brief Reset the ROOT object contained by the monitoring element
      \param me input Monitor Element
   */
  static void resetHisto( const MonitorElement* me ) {
    if( me ) {
      MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
      if( ob ) { 
	ob->Reset();
      }
    }
  }

 protected:
  EBMUtilsClient() {}

};

#endif

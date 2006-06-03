// $Id: EBMUtilsTasks.h,v 1.9 2006/05/23 09:06:49 benigno Exp $

/*!
  \file EBMUtilsTasks.h
  \brief Ecal Barrel Monitor Utils for Tasks
  \author B. Gobbo 
  \version $Revision: 1.9 $
  \date $Date: 2006/05/23 09:06:49 $
*/

#ifndef EBMUtilsTasks_H
#define EBMUtilsTasks_H

#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TROOT.h"

/*! \class EBMUtilsTasks
    \brief Utilities for Ecal Barrel Monitor Tasks
 */

class EBMUtilsTasks {

 public:

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
  EBMUtilsTasks() {}

};

#endif

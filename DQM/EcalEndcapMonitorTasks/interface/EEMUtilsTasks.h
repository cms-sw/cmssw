// $Id: EEMUtilsTasks.h,v 1.2 2007/03/26 17:34:07 dellaric Exp $

/*!
  \file EEMUtilsTasks.h
  \brief Ecal Barrel Monitor Utils for Tasks
  \author B. Gobbo
  \version $Revision: 1.2 $
  \date $Date: 2007/03/26 17:34:07 $
*/

#ifndef EEMUtilsTasks_H
#define EEMUtilsTasks_H

#include <string>
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TROOT.h"

/*! \class EEMUtilsTasks
    \brief Utilities for Ecal Barrel Monitor Tasks
 */

class EEMUtilsTasks {

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
  EEMUtilsTasks() {}

};

#endif

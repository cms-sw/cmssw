// $Id: EBMUtilsClient.cc,v 1.2 2007/02/20 11:01:18 benigno Exp $

/*!
  \file EBMUtilsClient.cc
  \brief Ecal Barrel Monitor Utils for Client
  \author B. Gobbo
  \version $Revision: 1.2 $
  \date $Date: 2007/02/20 11:01:18 $
*/

#include "DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h"

// ----------------------------------------------------------------------------------------------------
void EBMUtilsClient::resetHisto( const MonitorElement* me ) {
  if( me ) {
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if( ob ) {
      ob->Reset();
    }
  }
}

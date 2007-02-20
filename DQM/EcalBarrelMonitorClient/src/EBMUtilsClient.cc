// $Id: EBMUtilsClient.cc,v 1.1 2006/05/23 09:06:50 benigno Exp $

/*!
  \file EBMUtilsClient.cc
  \brief Ecal Barrel Monitor Utils for Client
  \author B. Gobbo 
  \version $Revision: 1.1 $
  \date $Date: 2006/05/23 09:06:50 $
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

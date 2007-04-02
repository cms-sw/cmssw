// $Id: EEMUtilsClient.cc,v 1.3 2007/03/26 17:35:05 dellaric Exp $

/*!
  \file EEMUtilsClient.cc
  \brief Ecal Barrel Monitor Utils for Client
  \author B. Gobbo
  \version $Revision: 1.3 $
  \date $Date: 2007/03/26 17:35:05 $
*/

#include "DQM/EcalEndcapMonitorClient/interface/EEMUtilsClient.h"

// ----------------------------------------------------------------------------------------------------
void EEMUtilsClient::resetHisto( const MonitorElement* me ) {
  if( me ) {
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if( ob ) {
      ob->Reset();
    }
  }
}

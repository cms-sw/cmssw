// $Id: $

/*!
  \file UtilsClient.cc
  \brief Ecal Monitor Utils for Client
  \author B. Gobbo
  \version $Revision: $
  \date $Date: $
*/

#include "DQM/EcalCommon/interface/UtilsClient.h"

// ----------------------------------------------------------------------------------------------------
void UtilsClient::resetHisto( const MonitorElement* me ) {
  if( me ) {
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if( ob ) {
      ob->Reset();
    }
  }
}

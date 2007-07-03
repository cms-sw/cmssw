#include "DQM/EcalPreshowerMonitorClient/interface/ESDQMUtils.h"

#include "TNamed.h"

void ESDQMUtils::resetME( const MonitorElement* me ) {

  if( me ) {
    MonitorElementT<TNamed>* ob = dynamic_cast<MonitorElementT<TNamed>*>( const_cast<MonitorElement*>(me) );
    if( ob ) {
      ob->Reset();
    }
  }

}

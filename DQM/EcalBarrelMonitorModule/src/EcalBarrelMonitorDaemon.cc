/*
 * \file EcalBarrelMonitorDaemon.cc
 *
 * $Date: 2005/11/17 08:43:49 $
 * $Revision: 1.1 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorDaemon.h>

DaqMonitorBEInterface* EcalBarrelMonitorDaemon::dbe_ = 0;

EcalBarrelMonitorDaemon::EcalBarrelMonitorDaemon(){

  cout << "Constructing a new EcalBarrelMonitorDaemon ... " << flush;

  // get hold of back-end interface
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) dbe_->setVerbose(1);

  //We put this here for the moment since there is no better place 
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

   cout << "done !" << endl;

}

EcalBarrelMonitorDaemon::~EcalBarrelMonitorDaemon(){
}

DaqMonitorBEInterface* EcalBarrelMonitorDaemon::dbe(){

  if ( dbe_ == 0 ) {
    new EcalBarrelMonitorDaemon();
  }

  cout << "EcalBarrelMonitorDaemon::dbe() returning a pointer to the DaqMonitorBEInterface" << endl;
  return dbe_;

}


/*
 * \file EcalEndcapMonitorDbModule.cc
 *
 * $Date: 2012/02/28 16:38:16 $
 * $Revision: 1.16 $
 * \author G. Della Ricca
 *
*/

#include <unistd.h>

#include <iostream>
#include <cmath>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IConnectionServiceConfiguration.h"

#include "CoralBase/Attribute.h"
#include "CoralKernel/Context.h"

#include "DQM/EcalBarrelMonitorDbModule/interface/MonitorElementsDb.h"

#include "DQM/EcalEndcapMonitorDbModule/interface/EcalEndcapMonitorDbModule.h"

EcalEndcapMonitorDbModule::EcalEndcapMonitorDbModule(const edm::ParameterSet& ps){

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  xmlFile_ = ps.getUntrackedParameter<std::string>( "xmlFile", "" );
  if ( xmlFile_.size() != 0 ) {
    std::cout << "Monitor Elements from DB xml source file is " << xmlFile_ << std::endl;
  }

  sleepTime_ = ps.getUntrackedParameter<int>( "sleepTime", 0 );
  std::cout << "Sleep time is " << sleepTime_ << " second(s)." << std::endl;

  // html output directory
  htmlDir_ = ps.getUntrackedParameter<std::string>("htmlDir", ".");

  if ( htmlDir_.size() != 0 ) {
    std::cout << " HTML output will go to"
	      << " htmlDir = " << htmlDir_ << std::endl;
  } else {
    std::cout << " HTML output is disabled" << std::endl;
  }

  ME_Db_ = new MonitorElementsDb( ps, xmlFile_ );

  if ( dqmStore_ ) dqmStore_->showDirStructure();

  icycle_ = 0;
  session_ = 0;

}

EcalEndcapMonitorDbModule::~EcalEndcapMonitorDbModule(){

  if ( ME_Db_ ) delete ME_Db_;

}

void EcalEndcapMonitorDbModule::beginJob(void){

  icycle_ = 0;

  if ( ME_Db_ ) ME_Db_->beginJob();

}

void EcalEndcapMonitorDbModule::endJob(void) {

  if ( ME_Db_ ) ME_Db_->endJob();

  std::cout << "EcalEndcapMonitorDbModule: endJob, icycle = " << icycle_ << std::endl;

}

void EcalEndcapMonitorDbModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  icycle_++;

  std::cout << "EcalEndcapMonitorDbModule: icycle = " << icycle_ << std::endl;

  try {
    coral::Context& context = coral::Context::instance();
    context.loadComponent("CORAL/Services/ConnectionService");
    context.loadComponent("CORAL/Services/EnvironmentAuthenticationService");
    coral::IHandle<coral::IConnectionService> connectionService = context.query<coral::IConnectionService>("CORAL/Services/ConnectionService");
    context.loadComponent("CORAL/RelationalPlugins/oracle");

    // Set configuration parameters
    coral::IConnectionServiceConfiguration& config = connectionService->configuration();
    config.setConnectionRetrialPeriod(1);
    config.setConnectionRetrialTimeOut(10);

    session_ = connectionService->connect("ECAL CondDB", coral::ReadOnly);

    if ( ME_Db_ ) ME_Db_->analyze(e, c, session_ );

  } catch (coral::Exception& e) {
    std::cerr << "CORAL Exception : " << e.what() << std::endl;
  } catch (std::exception& e) {
    std::cerr << "Standard C++ exception : " << e.what() << std::endl;
  }

  if ( htmlDir_.size() != 0 ) {

    ME_Db_->htmlOutput( htmlDir_ );

  }

  delete session_;

  sleep( sleepTime_ );

}


// Last commit: $Id: $

#include "DQM/SiStripCommissioningDbClients/interface/SiStripCommissioningOfflineDbClient.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommissioningDbClients/interface/FedCablingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/ApvTimingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/VpspScanHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/PedestalsHistosUsingDb.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <boost/cstdint.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include "TProfile.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineDbClient::SiStripCommissioningOfflineDbClient( const edm::ParameterSet& pset ) 
  : SiStripCommissioningOfflineClient(pset),
    uploadToDb_( pset.getUntrackedParameter<bool>("UploadToConfigDb",false) ),
    test_( pset.getUntrackedParameter<bool>("Test",false) )
{
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineDbClient::~SiStripCommissioningOfflineDbClient() {
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineDbClient::processHistos( DaqMonitorBEInterface* const bei,
							 const std::vector<std::string>& contents ) {

  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Processing histograms...";

  // Check pointer
  if ( !bei ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
    return;
  }

  // Create DB connection
  SiStripConfigDb* db = edm::Service<SiStripConfigDb>().operator->(); //@@ NOT GUARANTEED TO BE THREAD SAFE! 
  edm::LogWarning(mlCabling_) 
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Nota bene: using the SiStripConfigDb API"
    << " as a \"service\" does not presently guarantee"
    << " thread-safe behaviour!...";
  
  // Create corresponding "commissioning histograms" object 
  if      ( runType_ == sistrip::FED_CABLING )    { histos_ = new FedCablingHistosUsingDb( bei, db ); }
  else if ( runType_ == sistrip::APV_TIMING )     { histos_ = new ApvTimingHistosUsingDb( bei, db ); }
  else if ( runType_ == sistrip::OPTO_SCAN )      { histos_ = new OptoScanHistosUsingDb( bei, db ); }
  else if ( runType_ == sistrip::VPSP_SCAN )      { histos_ = new VpspScanHistosUsingDb( bei, db ); }
  else if ( runType_ == sistrip::PEDESTALS )      { histos_ = new PedestalsHistosUsingDb( bei, db ); }
  else if ( runType_ == sistrip::UNDEFINED_RUN_TYPE ) { histos_ = 0; }
  else if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) {
    histos_ = 0;
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningDbClient::" << __func__ << "]"
      << " Unknown run type!";
  }
  
  // Switch on test mode (inhibits DB upload)
  if ( test_ ) { 
    CommissioningHistosUsingDb* histos = 0;
    histos = dynamic_cast<CommissioningHistosUsingDb*>(histos_);
    if ( histos ) { histos->testOnly(true); }
    else { return; }
  }
  
  // Perform client "actions"
  if ( histos_ ) {
    histos_->extractHistograms( contents );
    histos_->histoAnalysis(true);
    if ( uploadToDb_ ) { histos_->uploadToConfigDb(); }
    //if ( createSummaryPlots_ ) { histos_->createSumaryPlots(); }
  }
  
  if ( saveSummaryPlots_ ) { bei->save("test.root"); }

}

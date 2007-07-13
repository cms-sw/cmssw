// Last commit: $Id: SiStripCommissioningOfflineDbClient.cc,v 1.4 2007/06/19 12:30:36 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/plugins/SiStripCommissioningOfflineDbClient.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommissioningDbClients/interface/FastFedCablingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/FedCablingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/ApvTimingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/VpspScanHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/PedestalsHistosUsingDb.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineDbClient::SiStripCommissioningOfflineDbClient( const edm::ParameterSet& pset ) 
  : SiStripCommissioningOfflineClient(pset),
    uploadToDb_( pset.getUntrackedParameter<bool>("DoNotUse",true) ),
    test_( /* note the "!" -> */ !pset.getUntrackedParameter<bool>("UploadToConfigDb",false) ),
    uploadFecSettings_( pset.getUntrackedParameter<bool>("UploadFecSettings",true) ),
    uploadFedSettings_( pset.getUntrackedParameter<bool>("UploadFedSettings",true) )
{
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Constructing object...";
  if ( test_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " ===> TEST only! No hardware configuration"
      << " settings will be uploaded to the DB...";
  }
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
void SiStripCommissioningOfflineDbClient::createCommissioningHistograms() {

  // Check pointer
  if ( histos_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " CommissioningHistogram object already exists!"
      << " Aborting...";
    return;
  } 

  // Check pointer to MUI
  if ( !mui_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!";
    return;
  }

  // Check pointer to BEI
  DaqMonitorBEInterface* bei = mui_->getBEInterface();
  if ( !bei ) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
    return;
  }

  // Create DB connection
  SiStripConfigDb* db = edm::Service<SiStripConfigDb>().operator->(); //@@ NOT GUARANTEED TO BE THREAD SAFE! 
  LogTrace(mlCabling_) 
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Nota bene: using the SiStripConfigDb API"
    << " as a \"service\" does not presently guarantee"
    << " thread-safe behaviour!...";
  
  // Check DB connection
  if ( !db ) {
    edm::LogWarning(mlCabling_) 
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb!"
      << " Aborting...";
    return;
  } 
  
  // Create corresponding "commissioning histograms" object 
  if ( runType_ == sistrip::FAST_CABLING ) { histos_ = new FastFedCablingHistosUsingDb( mui_, db ); }
  else if ( runType_ == sistrip::FED_CABLING ) { histos_ = new FedCablingHistosUsingDb( mui_, db ); }
  else if ( runType_ == sistrip::APV_TIMING ) { histos_ = new ApvTimingHistosUsingDb( mui_, db ); }
  else if ( runType_ == sistrip::OPTO_SCAN ) { histos_ = new OptoScanHistosUsingDb( mui_, db ); }
  else if ( runType_ == sistrip::VPSP_SCAN ) { histos_ = new VpspScanHistosUsingDb( mui_, db ); }
  else if ( runType_ == sistrip::PEDESTALS ) { histos_ = new PedestalsHistosUsingDb( mui_, db ); }
  else if ( runType_ == sistrip::UNDEFINED_RUN_TYPE ) { histos_ = 0; }
  else if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) {
    histos_ = 0;
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " Unknown run type!";
  }

  // 
  ApvTimingHistosUsingDb* temp = dynamic_cast<ApvTimingHistosUsingDb*>(histos_);
  if ( temp ) { 
    temp->uploadPllSettings( uploadFecSettings_ );
    temp->uploadFedSettings( uploadFedSettings_ );
  }
  
}

// -----------------------------------------------------------------------------
// Switch for "test mode" (inhibits actual database upload)
void SiStripCommissioningOfflineDbClient::testUploadToDb() {
  if ( test_ ) { 
    CommissioningHistosUsingDb* histos = 0;
    histos = dynamic_cast<CommissioningHistosUsingDb*>(histos_);
    if ( histos ) { histos->testOnly(true); }
    else {
      edm::LogWarning(mlDqmClient_) 
	<< "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
	<< " NULL pointer to CommissioningHistosUsingDb!";
    }
  }
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineDbClient::uploadToDb() {
  if ( uploadToDb_ ) { 
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " Uploading parameters to database...";
    if ( histos_ ) { histos_->uploadToConfigDb(); }
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " Uploaded parameters to database!";
  } else {
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " No database upload performed!";
  }
}

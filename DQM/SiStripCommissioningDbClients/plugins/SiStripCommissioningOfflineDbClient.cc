// Last commit: $Id: SiStripCommissioningOfflineDbClient.cc,v 1.26 2010/04/21 14:26:18 dstrom Exp $

#include "DQM/SiStripCommissioningDbClients/plugins/SiStripCommissioningOfflineDbClient.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommissioningDbClients/interface/FastFedCablingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/ApvTimingHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/OptoScanHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/VpspScanHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/PedestalsHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/PedsOnlyHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/NoiseHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/PedsFullNoiseHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/LatencyHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/FineDelayHistosUsingDb.h"
#include "DQM/SiStripCommissioningDbClients/interface/CalibrationHistosUsingDb.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineDbClient::SiStripCommissioningOfflineDbClient( const edm::ParameterSet& pset ) 
  : SiStripCommissioningOfflineClient(pset),
    uploadAnal_( pset.getUntrackedParameter<bool>("UploadAnalyses",false) ),
    uploadConf_( pset.getUntrackedParameter<bool>("UploadHwConfig",false) )
{
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Constructing object...";
  if ( !uploadConf_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " ===> TEST only! No hardware configurations"
      << " will be uploaded to the DB...";
  }
  if ( !uploadAnal_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " ===> TEST only! No analysis descriptions"
      << " will be uploaded to the DB...";
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
void SiStripCommissioningOfflineDbClient::createHistos( const edm::ParameterSet& pset, const edm::EventSetup& setup) {

  // Check pointer
  if ( histos_ ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " CommissioningHistogram object already exists!"
      << " Aborting...";
    return;
  } 

  // Check pointer to BEI
  // is this needed here? bei_ = edm::Service<DQMStore>().operator->();
  if ( !bei_ ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to DQMStore!";
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
    edm::LogError(mlCabling_) 
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb!"
      << " Aborting...";
    return;
  } 
  
  // Create corresponding "commissioning histograms" object 
  if      ( runType_ == sistrip::FAST_CABLING ) { histos_ = new FastFedCablingHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::APV_TIMING )   { histos_ = new ApvTimingHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::OPTO_SCAN )    { histos_ = new OptoScanHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::VPSP_SCAN )    { histos_ = new VpspScanHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::PEDESTALS )    { histos_ = new PedestalsHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::PEDS_ONLY )    { histos_ = new PedsOnlyHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::PEDS_FULL_NOISE ) { histos_ = new PedsFullNoiseHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::NOISE )        { histos_ = new NoiseHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::APV_LATENCY )  { histos_ = new LatencyHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::FINE_DELAY )   { histos_ = new FineDelayHistosUsingDb( pset, bei_, db ); }
  else if ( runType_ == sistrip::CALIBRATION ||
            runType_ == sistrip::CALIBRATION_DECO ||
            runType_ == sistrip::CALIBRATION_SCAN ||
            runType_ == sistrip::CALIBRATION_SCAN_DECO)
                                                { histos_ = new CalibrationHistosUsingDb( pset, bei_, db, runType_ ); }
  else if ( runType_ == sistrip::UNDEFINED_RUN_TYPE ) { 
    histos_ = 0; 
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " Undefined run type!";
    return;
  } else if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) {
    histos_ = 0;
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " Unknown run type!";
    return;
  }
  histos_->configure(pset,setup);

  CommissioningHistosUsingDb* tmp = dynamic_cast<CommissioningHistosUsingDb*>(histos_);
  if ( tmp ) { 
    tmp->doUploadConf( uploadConf_ ); 
    tmp->doUploadAnal( uploadAnal_ ); 
    std::stringstream ss;
    ss << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]" 
       << std::endl
       << " Uploading hardware configurations?    : " 
       << ( tmp->doUploadConf() ? "true" : "false" )
       << std::endl
       << " Uploading calibrations from analysis? : " 
       << ( tmp->doUploadAnal() ? "true" : "false" )
       << std::endl;
    edm::LogVerbatim(mlDqmClient_) << ss.str();
  } else {
    edm::LogError(mlDqmClient_) 
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to CommissioningHistosUsingDb!";
  }

}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineDbClient::uploadToConfigDb() {
  edm::LogVerbatim(mlDqmClient_)
    << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
    << " Uploading parameters to database...";
  CommissioningHistosUsingDb* tmp = dynamic_cast<CommissioningHistosUsingDb*>(histos_);
  if ( tmp ) { 
    tmp->uploadToConfigDb(); 
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " Uploaded parameters to database!";
  } else {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineDbClient::" << __func__ << "]"
      << " NULL pointer to CommissioningHistosUsingDb object!"
      << " Upload aborted!...";
  }
}

// Last commit: $Id: $

#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningOfflineClient.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include "TProfile.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineClient::SiStripCommissioningOfflineClient( const edm::ParameterSet& pset ) 
  : rootFile_( pset.getUntrackedParameter<std::string>("InputRootFile","source.root") ),
    xmlFile_( pset.getUntrackedParameter<std::string>("SummaryPlotXml","summary.xml") ),
    createSummaryPlots_( pset.getUntrackedParameter<bool>("CreateSummaryPlots",true) ),
    saveSummaryPlots_( pset.getUntrackedParameter<bool>("SaveSummaryPlots",true) ),
    runType_(sistrip::UNKNOWN_RUN_TYPE),
    view_(sistrip::UNKNOWN_VIEW),
    run_(0),
    map_(),
    plots_()
{
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineClient::~SiStripCommissioningOfflineClient() {
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::beginJob( const edm::EventSetup& setup ) {
  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Analyzing root file...";
  
  // Check if .root file can be opened
  ifstream root_file;
  root_file.open( rootFile_.c_str() );
  if( !root_file ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " The input Root file \"" << rootFile_
      << "\" could not be opened!"
      << " Please check the path and filename!";
    return;
  } else { root_file.close(); }

  // Check if .xml file can be opened
  ifstream xml_file;
  xml_file.open( xmlFile_.c_str() );
  if( !xml_file ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " The SummaryPlot XML file \"" << xmlFile_
      << "\" could not be opened!"
      << " Please check the path and filename!";
    return;
  } else { xml_file.close(); }
  
  // Retrieve BackEndInterface
  DaqMonitorBEInterface* bei = 0;
  bei = edm::Service<DaqMonitorBEInterface>().operator->();

  // Check pointer
  if ( !bei ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
    return;
  }

  // Open root file
  bei->setVerbose(0);
  bei->open(rootFile_,saveSummaryPlots_);
  
  // Open and parse "summary plot" xml file
  ConfigParser cfg;
  cfg.parseXML(xmlFile_);
  plots_ = cfg.summaryPlots(runType_);

  // Retrieve list of histograms
  std::vector<std::string> contents;
  CommissioningHistograms::getContents( bei, contents ); 
  
  // Extract commissioning task from adcontents
  runType_ = CommissioningHistograms::runType( bei, contents ); 
  
  // Check runType
  if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Unknown commissioning runType: " 
      << SiStripEnumsAndStrings::runType( runType_ );
    return;
  }

  // Some debug
  std::stringstream ss;
  ss << "[SiStripCommissioningOfflineClient::" << __func__ << "]" << std::endl
     << " Input root file       : " << rootFile_ << std::endl
     << " Run type              : " << SiStripEnumsAndStrings::runType( runType_ ) << std::endl
     << " Summary plot XML file : " << xmlFile_;
  edm::LogVerbatim(mlDqmClient_) << ss.str();

  // Process commissioning histograms
  processHistos( bei, contents );
  
  edm::LogVerbatim(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Finished analyzing .root file...";
  
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::processHistos( DaqMonitorBEInterface* const bei,
						       const std::vector<std::string>& contents ) {
  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Processing histograms...";

  // Check pointer
  if ( !bei ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
    return;
  }
  
  // Create corresponding "commissioning histograms" object 
  if      ( runType_ == sistrip::FED_CABLING )        { histos_ = new FedCablingHistograms( bei ); }
  else if ( runType_ == sistrip::APV_TIMING )         { histos_ = new ApvTimingHistograms( bei ); }
  else if ( runType_ == sistrip::OPTO_SCAN )          { histos_ = new OptoScanHistograms( bei ); }
  else if ( runType_ == sistrip::VPSP_SCAN )          { histos_ = new VpspScanHistograms( bei ); }
  else if ( runType_ == sistrip::PEDESTALS )          { histos_ = new PedestalsHistograms( bei ); }
  else if ( runType_ == sistrip::UNDEFINED_RUN_TYPE ) { histos_ = 0; }
  else if ( runType_ == sistrip::UNKNOWN_RUN_TYPE )   { 
    histos_ = 0;
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Unknown run type!";
  }
  
  // Perform client "actions"
  if ( histos_ ) {
    histos_->extractHistograms( contents );
    histos_->histoAnalysis(true);
    //if ( createSummaryPlots_ ) { histos_->createSumaryPlots(); }
  }
  
  if ( saveSummaryPlots_ ) { bei->save("test.root"); }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::analyze( const edm::Event& event, 
						 const edm::EventSetup& setup ) {
  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}

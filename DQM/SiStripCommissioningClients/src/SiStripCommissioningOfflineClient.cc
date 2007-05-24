// Last commit: $Id: SiStripCommissioningOfflineClient.cc,v 1.1 2007/04/04 07:16:16 bainbrid Exp $

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
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/cstdint.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "TProfile.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineClient::SiStripCommissioningOfflineClient( const edm::ParameterSet& pset ) 
  : mui_( new MonitorUIRoot() ),
    histos_(0),
    inputFiles_( pset.getUntrackedParameter< std::vector<std::string> >( "InputRootFiles", std::vector<std::string>() ) ),
    outputFileName_( pset.getUntrackedParameter<std::string>( "OutputRootFile", "" ) ),
    xmlFile_( pset.getUntrackedParameter<std::string>( "SummaryPlotXmlFile", "" ) ),
    createSummaryPlots_( false ),
    runType_(sistrip::UNKNOWN_RUN_TYPE),
    runNumber_(0),
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

  // Check for null pointer
  if ( !mui_ ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"
      << " Aborting...";
    return;
  }
   
  // Check if .root file can be opened
  std::vector<std::string>::const_iterator ifile = inputFiles_.begin();
  for ( ; ifile != inputFiles_.end(); ifile++ ) {
    ifstream root_file;
    root_file.open( ifile->c_str() );
    if( !root_file ) {
      edm::LogError(mlDqmClient_)
	<< "[SiStripCommissioningOfflineClient::" << __func__ << "]"
	<< " The input Root file \"" << *ifile
	<< "\" could not be opened!"
	<< " Please check the path and filename!"
	<< " Aborting...";
      return;
    } else { root_file.close(); }
  }
  
  // Check if .xml file can be opened
  if ( !xmlFile_.empty() ) {
    ifstream xml_file;
    xml_file.open( xmlFile_.c_str() );
    if( !xml_file ) {
      edm::LogError(mlDqmClient_)
	<< "[SiStripCommissioningOfflineClient::" << __func__ << "]"
	<< " The SummaryPlot XML file \"" << xmlFile_
	<< "\" could not be opened!"
	<< " Please check the path and filename!"
	<< " Aborting...";
      return;
    } else { 
      createSummaryPlots_ = true;
      xml_file.close(); 
    }
  }

  // Retrieve BEI and check for null pointer 
  DaqMonitorBEInterface* bei = mui_->getBEInterface();
  if ( !bei ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!"
      << " Aborting...";
    return;
  }
  bei->setVerbose(0);
  
  // Open root file(s) and create ME's
  std::vector<std::string>::const_iterator jfile = inputFiles_.begin();
  for ( ; jfile != inputFiles_.end(); jfile++ ) {
    bei->open( *jfile );
  }

  // Retrieve list of histograms
  std::vector<std::string> contents;
  mui_->getContents( contents ); 
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Found " << contents.size() 
    << " directories containing MonitorElements in "
    << inputFiles_.size() << " root files";

  // Extract run type from contents
  runType_ = CommissioningHistograms::runType( bei, contents ); 
  
  // Extract run number from contents
  runNumber_ = CommissioningHistograms::runNumber( bei, contents ); 
  
  // Check runType
  if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Unknown commissioning runType: " 
      << SiStripEnumsAndStrings::runType( runType_ );
    return;
  }
  
  // Open and parse "summary plot" xml file
  if ( createSummaryPlots_ ) {
    ConfigParser cfg;
    cfg.parseXML(xmlFile_);
    plots_ = cfg.summaryPlots(runType_);
    LogTrace(mlTest_)
      << "TEST3 " 
      << plots_.size() << " " 
      << SiStripEnumsAndStrings::runType( runType_ ) << " " 
      << cfg;
  } else {
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Null string for SummaryPlotXmlFile!"
      << " No summary plots will be created!";
  }
  
  // Some debug
  std::stringstream ss;
  ss << "[SiStripCommissioningOfflineClient::" << __func__ << "]" << std::endl
     << " Input root files      : ";
  if ( inputFiles_.empty() ) { ss << "(none)"; }
  else {
    std::vector<std::string>::const_iterator ifile = inputFiles_.begin();
    for ( ; ifile != inputFiles_.end(); ifile++ ) {
      if ( ifile != inputFiles_.begin() ) { 
	ss << std::setw(25) << std::setfill(' ') << ": ";
      }
      ss << "\"" << *ifile << "\"" << std::endl; 
    }
  }
  ss << " Run type              : \"" 
     << SiStripEnumsAndStrings::runType( runType_ ) << "\"" << std::endl
     << " Run number            :  " << runNumber_ << std::endl
     << " Summary plot XML file : ";
  if ( xmlFile_.empty() ) { ss << "(none)"; }
  else { ss << "\"" << xmlFile_ << "\""; }
  edm::LogVerbatim(mlDqmClient_) << ss.str();

  // Virtual method that creates CommissioningHistogram object
  createCommissioningHistograms();

  // Virtual method to switch on test mode for database uploads
  testUploadToDb();
  
  // Perform collation
  if ( histos_ ) { histos_->createCollations( contents ); }
    
  // Trigger update methods
  if ( mui_ ) { mui_->doSummary(); } //@@ temporary!
  
  // Perform analysis
  if ( histos_ ) { histos_->histoAnalysis( true ); }
  
  // Create summary plots
  if ( histos_ &&
       createSummaryPlots_ ) { 
    edm::LogVerbatim(mlTest_) 
      << "TEST1 " << plots_.size(); 
    std::vector<ConfigParser::SummaryPlot>::const_iterator iplot =  plots_.begin();
    for ( ; iplot != plots_.end(); iplot++ ) {
      edm::LogVerbatim(mlTest_) 
	<< "TEST2 " << *iplot; 
      histos_->createSummaryHisto( iplot->mon_,
				   iplot->pres_,
				   iplot->level_,
				   iplot->gran_ );
    }
  } else {
    edm::LogVerbatim(mlTest_) 
      << "TEST " << histos_ 
      << " " << createSummaryPlots_;
  }
  
  // Save client root file
  if ( histos_ ) { histos_->save( outputFileName_, runNumber_ ); }

  // Virtual method to trigger the database upload
  uploadToDb();
  
  // Remove all ME/CME objects and delete MUI
  if ( histos_ ) { histos_->remove(); }
  if ( mui_ ) { 
    if ( mui_->getBEInterface() ) { 
      mui_->getBEInterface()->setVerbose(0); 
    }
    delete mui_; 
  }
  
  edm::LogVerbatim(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Finished analyzing .root file...";
  
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::createCommissioningHistograms() {

  // Check pointer
  if ( histos_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " CommissioningHistogram object already exists!"
      << " Aborting...";
    return;
  } 

  // Check pointer to MUI
  if ( !mui_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!";
    return;
  }

  // Create "commissioning histograms" object 
  if      ( runType_ == sistrip::FED_CABLING )        { histos_ = new FedCablingHistograms( mui_ ); }
  else if ( runType_ == sistrip::APV_TIMING )         { histos_ = new ApvTimingHistograms( mui_ ); }
  else if ( runType_ == sistrip::OPTO_SCAN )          { histos_ = new OptoScanHistograms( mui_ ); }
  else if ( runType_ == sistrip::VPSP_SCAN )          { histos_ = new VpspScanHistograms( mui_ ); }
  else if ( runType_ == sistrip::PEDESTALS )          { histos_ = new PedestalsHistograms( mui_ ); }
  else if ( runType_ == sistrip::UNDEFINED_RUN_TYPE ) { histos_ = 0; }
  else if ( runType_ == sistrip::UNKNOWN_RUN_TYPE )   { 
    histos_ = 0;
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Unknown run type!";
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::analyze( const edm::Event& event, 
						 const edm::EventSetup& setup ) {
  if ( !(event.id().event()%10) ) {
    LogTrace(mlDqmClient_) 
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Empty event loop! User can kill job...";
  }
}


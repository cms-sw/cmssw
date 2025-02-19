// Last commit: $Id: SiStripCommissioningOfflineClient.cc,v 1.44 2010/01/04 16:47:06 lowette Exp $

#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningOfflineClient.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedsOnlyHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/NoiseHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/SamplingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <boost/cstdint.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include "TProfile.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
SiStripCommissioningOfflineClient::SiStripCommissioningOfflineClient( const edm::ParameterSet& pset ) 
  : bei_( edm::Service<DQMStore>().operator->() ),
    histos_(0),
    //inputFiles_( pset.getUntrackedParameter< std::vector<std::string> >( "InputRootFiles", std::vector<std::string>() ) ),
    outputFileName_( pset.getUntrackedParameter<std::string>( "OutputRootFile", "" ) ),
    collateHistos_( !pset.getUntrackedParameter<bool>( "UseClientFile", false ) ),
    analyzeHistos_( pset.getUntrackedParameter<bool>( "AnalyzeHistos", true ) ),
    xmlFile_( (pset.getUntrackedParameter<edm::FileInPath>( "SummaryXmlFile", edm::FileInPath() )).fullPath() ),
    createSummaryPlots_( false ),
    clientHistos_( false ), 
    uploadToDb_( false ), 
    runType_(sistrip::UNKNOWN_RUN_TYPE),
    runNumber_(0),
    map_(),
    plots_(),
    parameters_(pset)
{
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Constructing object...";
  setInputFiles( inputFiles_,
		 pset.getUntrackedParameter<std::string>( "FilePath" ),
		 pset.getUntrackedParameter<uint32_t>("RunNumber"), 
		 collateHistos_ );
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
void SiStripCommissioningOfflineClient::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Analyzing root file(s)...";

  // Check for null pointer
  if ( !bei_ ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " NULL pointer to DQMStore!"
      << " Aborting...";
    return;
  }
  bei_->setVerbose(0);

  // Check if .root file can be opened
  std::vector<std::string>::const_iterator ifile = inputFiles_.begin();
  for ( ; ifile != inputFiles_.end(); ifile++ ) {
    ifstream root_file;
    root_file.open( ifile->c_str() );
    if( !root_file.is_open() ) {
      edm::LogError(mlDqmClient_)
	<< "[SiStripCommissioningOfflineClient::" << __func__ << "]"
	<< " The input root file \"" << *ifile
	<< "\" could not be opened!"
	<< " Please check the path and filename!";
    } else { 
      root_file.close(); 
      std::string::size_type found = ifile->find(sistrip::dqmClientFileName_);
      if ( found != std::string::npos && clientHistos_ ) {
	edm::LogError(mlDqmClient_)
	  << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
	  << " The input root files appear to be a mixture"
	  << " of \"Source\" and \"Client\" files!"
	  << " Aborting...";
	return;
      }
      if ( found != std::string::npos && inputFiles_.size() != 1 ) {
	edm::LogError(mlDqmClient_)
	  << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
	  << " There appear to be multiple input \"Client\" root files!"
	  << " Aborting...";
	return;
      }
      if ( found != std::string::npos ) { clientHistos_ = true; }
    }
  }
  if ( clientHistos_ && inputFiles_.size() == 1 ) {
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Collated histograms found in input root file \""
      << inputFiles_[0] << "\"";
  }
  
  // Check if .xml file can be opened
  if ( !xmlFile_.empty() ) {
    ifstream xml_file;
    xml_file.open( xmlFile_.c_str() );
    if( !xml_file.is_open() ) {
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

  // Open root file(s) and create ME's
  if ( inputFiles_.empty() ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " No input root files specified!";
    return;
  }

  edm::LogVerbatim(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Opening root files. This may take some time!...";
  std::vector<std::string>::const_iterator jfile = inputFiles_.begin();
  for ( ; jfile != inputFiles_.end(); jfile++ ) {
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Opening root file \"" << *jfile
      << "\"... (This may take some time.)";
    if ( clientHistos_ ) {
      bei_->open( *jfile, false, sistrip::collate_, "" );
    } else { 
      bei_->open( *jfile, false, "SiStrip", sistrip::collate_ );
    }
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Opened root file \"" << *jfile << "\"!";
  }
  edm::LogVerbatim(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Opened " << inputFiles_.size() << " root files!";
  
  // Retrieve list of histograms
  std::vector<std::string> contents;
  bei_->getContents( contents ); 
  
  // If using client file, remove "source" histograms from list
  if ( clientHistos_ ) {
    std::vector<std::string> temp;
    std::vector<std::string>::iterator istr = contents.begin();
    for ( ; istr != contents.end(); istr++ ) {
      if ( istr->find(sistrip::collate_) != std::string::npos ) { 
	temp.push_back( *istr );
      }
    }
    contents.clear();
    contents = temp;
  }
  
  // Some debug
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Found " << contents.size() 
    << " directories containing MonitorElements";
  
  // Some more debug
  if (0) {
    std::stringstream ss;
    ss << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
       << " Directories found: " << std::endl;
    std::vector<std::string>::iterator istr = contents.begin();
    for ( ; istr != contents.end(); istr++ ) { ss << " " << *istr << std::endl; }
    LogTrace(mlDqmClient_) << ss.str();
  }
  
  // Extract run type from contents
  runType_ = CommissioningHistograms::runType( bei_, contents ); 
  
  // Extract run number from contents
  runNumber_ = CommissioningHistograms::runNumber( bei_, contents ); 

  // Copy custom information to the collated structure
  CommissioningHistograms::copyCustomInformation( bei_, contents );

  // Check runType
  if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) { 
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Unknown commissioning runType: " 
      << SiStripEnumsAndStrings::runType( runType_ )
      << " and run number is " << runNumber_;
    return;
  } else {
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Run type is " 
      << SiStripEnumsAndStrings::runType( runType_ )
      << " and run number is " << runNumber_;
  }
  
  // Open and parse "summary plot" xml file
  if ( createSummaryPlots_ ) {
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Parsing summary plot XML file...";
    SummaryPlotXmlParser xml_file;
    xml_file.parseXML(xmlFile_);
    plots_ = xml_file.summaryPlots(runType_);
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Parsed summary plot XML file and found " 
      << plots_.size() << " plots defined!";
  } else {
    edm::LogWarning(mlDqmClient_)
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
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Creating CommissioningHistogram object...";
  createHistos(parameters_, setup); 
  if ( histos_ ) {
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Created CommissioningHistogram object!";
  } else {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " NULL pointer to CommissioningHistogram object!"
      << " Aborting...";
    return;
  }
  
  // Perform collation
  if ( histos_ ) { 
    histos_->extractHistograms( contents ); 
  }
  
  // Perform analysis
  if ( analyzeHistos_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Analyzing histograms...";
    if ( histos_ ) { histos_->histoAnalysis( true ); }
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Analyzed histograms!";
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " No histogram analysis performed!";
  }
  
  // Create summary plots
  if ( createSummaryPlots_ ) { 
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Generating summary plots...";
    std::vector<SummaryPlot>::const_iterator iplot =  plots_.begin();
    for ( ; iplot != plots_.end(); iplot++ ) {
      if ( histos_ ) { 
	histos_->createSummaryHisto( iplot->monitorable(),
				     iplot->presentation(),
				     iplot->level(),
				     iplot->granularity() );
      }
    }
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Generated summary plots!";
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " No summary plots generated!";
  }
  
  // Save client root file
  if ( histos_ ) {
    bool save = parameters_.getUntrackedParameter<bool>( "SaveClientFile", true );
    if ( save ) { histos_->save( outputFileName_, runNumber_ ); }
    else {
      edm::LogVerbatim(mlDqmClient_)
	<< "[SiStripCommissioningOfflineClient::" << __func__ << "]"
	<< " Client file not saved!";
    }
  }
  
  // Virtual method to trigger the database upload
  uploadToConfigDb();
  
  // Print analyses
  if ( histos_ ) { 
    histos_->printAnalyses(); 
    histos_->printSummary(); 
  }
  
  // Remove all ME/CME objects
  if ( histos_ ) { histos_->remove(); } 
  
  edm::LogVerbatim(mlDqmClient_)
    << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
    << " Finished analyzing root file(s)...";
  
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

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::endJob() {}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::createHistos( const edm::ParameterSet& pset, const edm::EventSetup& setup ) {
  
  // Check pointer
  if ( histos_ ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " CommissioningHistogram object already exists!"
      << " Aborting...";
    return;
  } 
  
  // Check pointer to BEI
  if ( !bei_ ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " NULL pointer to DQMStore!";
    return;
  }

  // Create "commissioning histograms" object 
  if      ( runType_ == sistrip::FAST_CABLING )         { histos_ = new FastFedCablingHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::FED_CABLING )          { histos_ = new FedCablingHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::APV_TIMING )           { histos_ = new ApvTimingHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::OPTO_SCAN )            { histos_ = new OptoScanHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::VPSP_SCAN )            { histos_ = new VpspScanHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::PEDESTALS )            { histos_ = new PedestalsHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::PEDS_ONLY )            { histos_ = new PedsOnlyHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::NOISE )                { histos_ = new NoiseHistograms( pset, bei_ ); }
  else if ( runType_ == sistrip::APV_LATENCY      ||
            runType_ == sistrip::FINE_DELAY           ) { histos_ = new SamplingHistograms( pset, bei_,runType_ ); }
  else if ( runType_ == sistrip::CALIBRATION      ||
	    runType_ == sistrip::CALIBRATION_DECO ||
	    runType_ == sistrip::CALIBRATION_SCAN ||
	    runType_ == sistrip::CALIBRATION_SCAN_DECO) { histos_ = new CalibrationHistograms( pset, bei_,runType_ ); }
  else if ( runType_ == sistrip::UNDEFINED_RUN_TYPE   ) { 
    histos_ = 0; 
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Undefined run type!";
    return;
  } else if ( runType_ == sistrip::UNKNOWN_RUN_TYPE )   { 
    histos_ = 0;
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Unknown run type!";
    return;
  }
  histos_->configure(pset,setup);
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningOfflineClient::setInputFiles( std::vector<std::string>& files,
						       const std::string path,
						       uint32_t run_number, 
						       bool collate_histos ) {

  std::string runStr;
  std::stringstream ss;
  ss << std::setfill('0') << std::setw(8) << run_number;
  runStr = ss.str();
  
  std::string nameStr = "";
  if ( !collate_histos ) { nameStr = "SiStripCommissioningClient_"; }
  else { nameStr = "SiStripCommissioningSource_"; }

  LogTrace("TEST") << " runStr " << runStr;

  // Open directory
  DIR* dp;
  struct dirent* dirp;
  if ( (dp = opendir(path.c_str())) == NULL ) {
    edm::LogError(mlDqmClient_) 
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " Error locating directory \"" << path
      << "\". No such directory!";
    return;
  }

  // Find compatible files
  while ( (dirp = readdir(dp)) != NULL ) {
    std::string fileName(dirp->d_name);
    bool goodName = ( fileName.find(nameStr) != std::string::npos );
    bool goodRun  = ( fileName.find(runStr) != std::string::npos );
    bool rootFile = ( fileName.find(".root") != std::string::npos );
    //bool rootFile = ( fileName.rfind(".root",5) == fileName.size()-5 );
    if ( goodName && goodRun && rootFile ) {
      std::string entry = path;
      entry += "/";
      entry += fileName;
      files.push_back(entry);
    }
  }
  closedir(dp);

  // Some debug  
  if ( !collate_histos && files.size() > 1 ) {
    std::stringstream ss;
    ss << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
       << " Found more than one client file!";
    std::vector<std::string>::const_iterator ifile = files.begin();
    std::vector<std::string>::const_iterator jfile = files.end();
    for ( ; ifile != jfile; ++ifile ) { ss << std::endl << *ifile; }
    edm::LogError(mlDqmClient_) << ss.str();
  } else if ( files.empty() ) {
    edm::LogError(mlDqmClient_)
      << "[SiStripCommissioningOfflineClient::" << __func__ << "]"
      << " No input files found!" ;
  }

}


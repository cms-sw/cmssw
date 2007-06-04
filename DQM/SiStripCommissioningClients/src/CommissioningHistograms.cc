#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui,
						  const sistrip::RunType& task ) 
  : mui_(mui),
    bei_(0),
    histos_(),
    task_(task)
{
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Constructing object...";

  // MonitorUserInterface
  if ( mui_ ) { bei_ = mui_->getBEInterface(); }
  else {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!";
  }
  
  // DaqMonitorBEInterface
  if ( !bei_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
  }
  
  clearHistosMap();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( DaqMonitorBEInterface* bei,
						  const sistrip::RunType& task ) 
  : mui_(0),
    bei_(bei),
    histos_(),
    task_(task)
{
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Constructing object...";

  // DaqMonitorBEInterface
  if ( !bei_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
  }
  
  clearHistosMap();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Destructing object...";
  clearHistosMap();
  //@@ do not delete MUI or BEI ptrs!
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::Histo::print( std::stringstream& ss ) const {
  ss << " [Histo::" << __func__ << "]" << std::endl
     << " Histogram title   : " << title_ << std::endl
     << " MonitorElement*   : 0x" 
     << std::hex
     << std::setw(8) << std::setfill('0') << me_ << std::endl
     << std::dec
     << " CollateME*        : 0x" 
     << std::hex
     << std::setw(8) << std::setfill('0') << cme_ << std::endl
     << std::dec;
}

// -----------------------------------------------------------------------------
// Temporary fix: builds a list of histogram directories
void CommissioningHistograms::getContents( DaqMonitorBEInterface* const bei,
					   std::vector<std::string>& contents ) {
  
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Building histogram list...";

  contents.clear();

  if ( !bei ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
  }

  bei->cd();
  std::vector<MonitorElement*> mons;
  mons = bei->getAllContents( bei->pwd() );
  std::vector<MonitorElement*>::const_iterator iter = mons.begin();
  for ( ; iter != mons.end(); iter++ ) {
    std::vector<std::string>::iterator istr = contents.begin();
    while ( istr != contents.end()  ) {
      //if ( istr->find( (*iter)->getPathname() ) != std::string::npos ) {
      if ( std::string((*iter)->getPathname()+"/:") == *istr ) {
	// 	if ( istr->find( (*iter)->getName() ) == std::string::npos ) {
	// 	  std::string temp( "," + (*iter)->getName() ); 
	// 	  (*istr) += temp;
	// 	}
	break;
      }
      istr++;
    }
    if ( istr == contents.end() ) { 
      std::string temp = (*iter)->getPathname() + "/:"; // + (*iter)->getName();
      contents.push_back( temp ); 
    }
  }
  
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Found " << contents.size() << " directories!";

  if ( contents.empty() ) { 
    edm::LogWarning(mlDqmClient_) 
      << "[CommissioningHistograms::" << __func__ << "]"
      << " No directories found when building list!";
  }
  
}

// -----------------------------------------------------------------------------
/** Extract run type string from "added contents". */
sistrip::RunType CommissioningHistograms::runType( DaqMonitorBEInterface* const bei,
						   const std::vector<std::string>& contents ) {
  
  // Check if histograms present
  if ( contents.empty() ) { 
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Found no histograms!";
    return sistrip::UNKNOWN_RUN_TYPE; 
  }
  
  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {

    // Extract source directory path 
    std::string source_dir = istr->substr( 0, istr->find(":") );
    
    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path( source_dir );
    std::string client_dir = path.path();
    //std::string client_dir = sistrip::root_ + "/"; //@@ 
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 

//     LogTrace(mlDqmClient_)
//       << "[CommissioningHistograms::" << __func__ << "]"
//       << " source dir: " << source_dir
//       << " client dir: " << client_dir;
    
    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents( source_dir );
    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      if ( !(*ime) ) {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " NULL pointer to MonitorElement!";
	continue;
      }

      // Search for run type in string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find( sistrip::taskId_ );

      // Extract commissioning task from string 
      if ( pos != std::string::npos ) { 
	std::string value = title.substr( pos+sistrip::taskId_.size()+1, std::string::npos ); 
	if ( !value.empty() ) { 
	  edm::LogVerbatim(mlDqmClient_)
	    << "[CommissioningHistograms::" << __func__ << "]"
	    << " Found string \"" <<  title.substr(pos,std::string::npos)
	    << "\" with value \"" << value << "\"";
	  if ( !(bei->get(client_dir+"/"+title.substr(pos,std::string::npos))) ) { 
	    bei->cd(client_dir);
	    bei->bookString( title.substr(pos,std::string::npos), value ); 
	  }
	  return SiStripEnumsAndStrings::runType( value ); 
	}
      }

    }

    istr++;
    
  }
  return sistrip::UNKNOWN_RUN_TYPE;
}

// -----------------------------------------------------------------------------
//
uint32_t CommissioningHistograms::runNumber( DaqMonitorBEInterface* const bei,
					     const std::vector<std::string>& contents ) {
  
  // Check if histograms present
  if ( contents.empty() ) { 
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Found no histograms!";
    return 0; 
  }
  
  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {
    
    // Extract source directory path 
    std::string source_dir = istr->substr( 0, istr->find(":") );
    
    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path( source_dir );
    std::string client_dir = path.path();
    //std::string client_dir = sistrip::root_ + "/"; //@@ 
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 
    
//     LogTrace(mlDqmClient_)
//       << "[CommissioningHistograms::" << __func__ << "]"
//       << " source dir: " << source_dir
//       << " client dir: " << client_dir;
    
    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents( source_dir );
    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      if ( !(*ime) ) {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " NULL pointer to MonitorElement!";
	continue;
      }

      // Search for run type in string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find( sistrip::runNumber_ );
      
      // Extract run number from string 
      if ( pos != std::string::npos ) { 
	std::string value = title.substr( pos+sistrip::runNumber_.size()+1, std::string::npos ); 
	if ( !value.empty() ) { 
	  edm::LogVerbatim(mlDqmClient_)
	    << "[CommissioningHistograms::" << __func__ << "]"
	    << " Found string \"" <<  title.substr(pos,std::string::npos)
	    << "\" with value \"" << value << "\"";
	  if ( !(bei->get(client_dir+"/"+title.substr(pos,std::string::npos))) ) { 
	    bei->cd(client_dir);
	    bei->bookString( title.substr(pos,std::string::npos), value ); 
	  }
	  uint32_t run;
	  std::stringstream ss;
	  ss << value;
	  ss >> std::dec >> run;
	  return run; 
	}
      }

    }

    istr++;
    
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::extractHistograms( const std::vector<std::string>& contents ) {
  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Extracting available histograms...";
  
  // Check pointer
  if ( !bei_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DaqMonitorBEInterface!";
    return;
  }
  
  // Check list of histograms
  if ( contents.empty() ) { return; }
  
  // Iterate through list of histograms
  std::vector<std::string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Ignore Collector/EvF directories (in case of client root file)
    if ( idir->find("Collector") == std::string::npos &&
 	 idir->find("EvF") == std::string::npos &&
	 idir->find("FU") == std::string::npos ) { continue; }
    
    // Extract source directory path 
    std::string source_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey path( source_dir );

    // Check path is valid 
    if ( path.granularity() == sistrip::FEC_SYSTEM ||
	 path.granularity() == sistrip::UNKNOWN_GRAN ||
	 path.granularity() == sistrip::UNDEFINED_GRAN ) { 
      //LogTrace(mlDqmClient_)
      //<< "[CommissioningHistograms::" << __func__ << "]"
      //<< " Unexpected granularity for path: " << path;
      continue; 
    }
    
    // Generate corresponding client path (removing trailing "/")
    std::string client_dir = SiStripFecKey(path.key()).path();
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 

    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei_->getContents( source_dir );
    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Check granularity
      SiStripHistoTitle title( (*ime)->getName() );
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = SiStripFecKey::lldChan( title.channel() );
      } else if ( title.granularity() == sistrip::UNKNOWN_GRAN || 
		  title.granularity() == sistrip::UNDEFINED_GRAN ) {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " Unexpected granularity for histogram title: "
	  << title << " found in path " << path;
      } else {
	channel = title.channel();
      }
      
      // Build FEC key
      uint32_t fec_key = SiStripFecKey( path.fecCrate(),
					path.fecSlot(),
					path.fecRing(),
					path.ccuAddr(),
					path.ccuChan(),
					channel ).key();
      
      // Fill FED-FEC map
      mapping_[title.keyValue()] = fec_key;
      
      // Find histogram in map
      Histo* histo = 0;
      HistosMap::iterator ihistos = histos_.find( fec_key );
      if ( ihistos != histos_.end() ) { 
	Histos::iterator ihis = ihistos->second.begin();
	while ( !histo && ihis < ihistos->second.end() ) {
	  if ( (*ime)->getName() == (*ihis)->title_ ) { histo = *ihis; }
	  ihis++;
	}
      }
      
      // Insert histogram into map if missing
      if ( !histo ) {
	histos_[fec_key].push_back( new Histo() );
	histo = histos_[fec_key].back();
	histo->title_ = (*ime)->getName();
	// If histogram present in client directory, add to map
	if ( source_dir.find("Collector") == std::string::npos &&
	     source_dir.find("EvF") == std::string::npos ) { 
	  histo->me_ = bei_->get( client_dir +"/"+(*ime)->getName() ); 
	  if ( !histo->me_ ) { 
	    edm::LogWarning(mlDqmClient_)
	      << "[CommissioningHistograms::" << __func__ << "]"
	      << " NULL pointer to MonitorElement!";
	  }
	}
      }
      
    }
  }

  //printHistosMap();
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const std::vector<std::string>& contents ) {
  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Creating collated histograms...";

  // Check pointer
  if ( !mui_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!";
    return;
  }
  
  // Check list of histograms
  if ( contents.empty() ) { return; }
  
  // Iterate through list of histograms
  std::vector<std::string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Ignore directories on client side
    if ( idir->find("Collector") == std::string::npos &&
 	 idir->find("EvF") == std::string::npos &&
	 idir->find("FU") == std::string::npos ) { continue; }
    
    // Extract source directory path 
    std::string source_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey path( source_dir );

//     std::stringstream ss;
//     ss << ">>> TEST1 " << std::endl
//        << " source_dir: " << source_dir << std::endl
//        << " path: " << path;
//     LogTrace(mlDqmClient_) << ss.str();
    
    // Check path is valid
    if ( path.granularity() == sistrip::FEC_SYSTEM ||
	 path.granularity() == sistrip::UNKNOWN_GRAN ||
	 path.granularity() == sistrip::UNDEFINED_GRAN ) { 
      //LogTrace(mlDqmClient_)
      //<< "[CommissioningHistograms::" << __func__ << "]"
      //<< " Unexpected granularity for path: " << path;
      continue; 
    }
    
    // Generate corresponding client path (removing trailing "/")
    std::string client_dir = SiStripFecKey(path.key()).path();
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 
    
//     std::stringstream sss;
//     sss << ">>> TEST2 " << std::endl
//        << " client_dir: " << client_dir;
//     LogTrace(mlDqmClient_) << sss.str();

    // Retrieve MonitorElements from pwd directory
    mui_->setCurrentFolder( source_dir );
    std::vector<std::string> me_list = mui_->getMEs();

    // Iterate through MonitorElements and create CMEs
    std::vector<std::string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve histogram title
      SiStripHistoTitle title( *ime );

      // Ignore summary plots
      //@@ what?? if ( 1 ) {;}
      
      // Check granularity
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = SiStripFecKey::lldChan( title.channel() );
      } else if ( title.granularity() == sistrip::UNKNOWN_GRAN || 
		  title.granularity() == sistrip::UNDEFINED_GRAN ) {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " Unexpected granularity for histogram title: "
	  << title << " found in path " << path;
      } else {
	channel = title.channel();
      }
      
      // Build FEC key
      uint32_t fec_key = SiStripFecKey( path.fecCrate(),
					path.fecSlot(),
					path.fecRing(),
					path.ccuAddr(),
					path.ccuChan(),
					channel ).key();
      
      // Fill FED-FEC map
      mapping_[title.keyValue()] = fec_key;

      // Find CollateME in histos map
      Histo* histo = 0;
      HistosMap::iterator ihistos = histos_.find( fec_key );
      if ( ihistos != histos_.end() ) { 
	Histos::iterator ihis = ihistos->second.begin();
	while ( !histo && ihis < ihistos->second.end() ) {
	  if ( (*ime) == (*ihis)->title_ ) { histo = *ihis; }
	  ihis++;
	}
      }
      
      // Create CollateME if it doesn't exist
      if ( !histo ) {

// 	LogTrace(mlTest_) 
// 	  << ">>> TEST " << std::endl
// 	  << " fec_key: " << SiStripFecKey( fec_key );

	// Retrieve ME pointer
	MonitorElement* me = mui_->get( mui_->pwd()+"/"+(*ime) );
	
// 	std::stringstream ssss;
// 	ssss << ">>> TEST3 " << std::endl
// 	   << " pwd: " << mui_->pwd()+"/"+(*ime);
// 	LogTrace(mlDqmClient_) << ssss.str();
	
	// Check if profile or 1D
	TProfile* prof = ExtractTObject<TProfile>().extract( me );
	TH1F* his = ExtractTObject<TH1F>().extract( me );

	// Create CollateME and extract pointer to ME
	if ( prof || his ) { 
	  histos_[fec_key].push_back( new Histo() );
	  histo = histos_[fec_key].back();
	  histo->title_ = *ime;
	  if ( prof ) {
	    prof->SetErrorOption("s"); //@@ necessary?
	    histo->cme_ = mui_->collateProf( (*ime), (*ime), client_dir ); 
	  } else if ( his ) {
	    histo->cme_ = mui_->collate1D( (*ime), (*ime), client_dir ); 
	  }
	  if ( histo->cme_ ) { 
	    mui_->add( histo->cme_, mui_->pwd()+"/"+(*ime) );
	    histo->me_ = histo->cme_->getMonitorElement(); 
	  }
	}
	
      }

      // Add to CollateME if found in histos map
      HistosMap::iterator jhistos = histos_.find( fec_key );
      if ( jhistos != histos_.end() ) { 
	Histos::iterator ihis = jhistos->second.begin();
	while ( ihis < jhistos->second.end() ) {
	  if ( (*ime) == (*ihis)->title_ ) { 
	    if ( (*ihis)->cme_ ) {
	      mui_->add( (*ihis)->cme_, mui_->pwd()+"/"+(*ime) );
	    }
	    break; 
	  }
	  ihis++;
	}
      }
      
    }
  }
  
  //printHistosMap();

  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Created collated histograms!";
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::clearHistosMap() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Clearing histogram map...";
  HistosMap::iterator ihistos = histos_.begin();
  for ( ; ihistos != histos_.end(); ihistos++ ) {
    Histos::iterator ihisto = ihistos->second.begin();
    for ( ; ihisto != ihistos->second.end(); ihisto++ ) {
      if ( *ihisto ) { delete *ihisto; }
    }
    ihistos->second.clear();
  }
  histos_.clear();
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printHistosMap() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Printing histogram map, which has "
    << histos_.size() << " entries...";
  HistosMap::const_iterator ihistos = histos_.begin();
  for ( ; ihistos != histos_.end(); ihistos++ ) {
    std::stringstream ss;
    ss << " Found " << ihistos->second.size()
       << " histograms for FEC key: "
       << SiStripFecKey(ihistos->first) << std::endl;
    Histos::const_iterator ihisto = ihistos->second.begin();
    for ( ; ihisto != ihistos->second.end(); ihisto++ ) {
      if ( *ihisto ) { (*ihisto)->print(ss); }
      else { ss << " NULL pointer to Histo object!"; }
    }
    LogTrace(mlDqmClient_) << ss.str();
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHisto( const sistrip::Monitorable& histo, 
						  const sistrip::Presentation& type, 
						  const std::string& directory,
						  const sistrip::Granularity& gran ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " (Derived) implementation to come..."; 
}

// -----------------------------------------------------------------------------
/** Wraps other createSummaryHisto() method. */
void CommissioningHistograms::createSummaryHisto( pair<sistrip::Monitorable,
						  sistrip::Presentation> summ0, 
						  pair<std::string,
						  sistrip::Granularity> summ1 ) {
  createSummaryHisto( summ0.first, summ0.second, summ1.first, summ1.second );
}

// -----------------------------------------------------------------------------
// 
TH1* CommissioningHistograms::histogram( const sistrip::Monitorable& mon, 
					 const sistrip::Presentation& pres, 
					 const sistrip::View& view,
					 const std::string& directory,
					 const uint32_t& xbins ) {
  
  // Remember pwd 
  std::string pwd = mui_->pwd();
  mui_->setCurrentFolder( directory );

  // Book new histogram
  std::string name = SummaryGenerator::name( task_, mon, pres, view, directory );
  MonitorElement* me = mui_->get( mui_->pwd() + "/" + name );
  if ( !me ) { 
    if ( pres == sistrip::SUMMARY_HISTO ) { 
      me = mui_->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( pres == sistrip::SUMMARY_1D ) { 
      me = mui_->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( pres == sistrip::SUMMARY_2D ) { 
      me = mui_->getBEInterface()->book2D( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else if ( pres == sistrip::SUMMARY_PROF ) { 
      me = mui_->getBEInterface()->bookProfile( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else { me = 0; 
    }
  }
  TH1F* summary = ExtractTObject<TH1F>().extract( me ); 
  
  // Return to pwd
  mui_->setCurrentFolder( pwd );
  
  return summary;
  
}



// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::remove( std::string pattern ) {
  
  if ( !mui_ ) { 
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
  mui_->getBEInterface()->setVerbose(0);

  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Removing histograms...";
  
  if ( !pattern.empty() ) {
    
    if ( mui_->getBEInterface()->dirExists(pattern) ) {
      mui_->getBEInterface()->rmdir(pattern); 
    }
    
    LogTrace(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Removing directories (and MonitorElements"
      << " therein) that match the pattern \""
      << pattern << "\"";
    
  } else {
    
    mui_->getBEInterface()->cd();
    mui_->getBEInterface()->removeContents(); 
    
    if( mui_->getBEInterface()->dirExists("Collector") ) {
      mui_->getBEInterface()->rmdir("Collector");
    }
    if( mui_->getBEInterface()->dirExists("EvF") ) {
      mui_->getBEInterface()->rmdir("EvF");
    }
    if( mui_->getBEInterface()->dirExists("SiStrip") ) {
      mui_->getBEInterface()->rmdir("SiStrip");
    }

    LogTrace(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Removing all directories (and MonitorElements therein)";
    
  }

  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Removed histograms!";

  mui_->getBEInterface()->setVerbose(1);

}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::save( std::string& path,
				    uint32_t run_number ) {
  
  if ( !mui_ ) { 
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }

  // Construct path and filename
  std::stringstream ss; 

  if ( !path.empty() ) { 

    ss << path; 
    if ( ss.str().find(".root") == std::string::npos ) { ss << ".root"; }

  } else {

    // Retrieve SCRATCH directory
    std::string scratch = "SCRATCH";
    std::string dir = "";
    if ( getenv(scratch.c_str()) != NULL ) { 
      dir = getenv(scratch.c_str()); 
    }
    
    // Add directory path 
    if ( !dir.empty() ) { ss << dir; }
    else { ss << "/tmp/"; }
    
    // Add filename with run number and ".root" extension
    ss << sistrip::dqmClientFileName_ << "_" 
       << std::setfill('0') << std::setw(7) << run_number
       << ".root";
    
  }
  
  // Save file with appropriate filename
  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Saving histograms to root file \""
    << ss.str() << "\"... (This may take some time!)";
  path = ss.str();
  mui_->save( path ); 
  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Saved histograms to root file!";
  
}

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
  if ( contents.empty() ) { return sistrip::UNKNOWN_RUN_TYPE; }
  
  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {

    // Extract source directory path 
    std::string source_dir = istr->substr( 0, istr->find(":") );
    
    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path( source_dir );
    //std::string client_dir = path.path();
    std::string client_dir = sistrip::root_ + "/"; //@@ 
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 
    
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

      // Search for "commissioning task" std::string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find( sistrip::taskId_ );

      // Extract commissioning task from std::string 
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
/** */
void CommissioningHistograms::extractHistograms( const std::vector<std::string>& contents ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Extracting available histograms...";

  // Check pointer
  if ( mui_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NON-ZERO pointer to MonitorUserInterface!"
      << " Should use createCollactions() instead!";
    return;
  }

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
    if ( idir->find("Collector") != std::string::npos ||
	 idir->find("EvF") != std::string::npos ) { continue; }
    
    // Extract source directory path 
    std::string source_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey path( source_dir );

    // Check path is valid to level of a module
    //if ( path.granularity() != sistrip::CCU_CHAN ) { continue; } //@@
    
    // Generate corresponding client path (removing trailing "/")
    std::string client_dir = path.path();
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 

    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei_->getContents( source_dir );
    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve granularity from histogram title (necessary?)
      SiStripHistoTitle title( (*ime)->getName() );
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = SiStripFecKey::lldChan(title.channel())-1; //@@ temporary!!!
      } else if ( title.granularity() == sistrip::LLD_CHAN ) {
	channel = title.channel();
      } else {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " Unexpected histogram granularity: "
	  << title.granularity();
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
	while ( ihis < ihistos->second.end() ) {
	  if ( (*ime)->getName() == (*ihis)->title_ ) { break; }
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

  printHistosMap();
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const std::vector<std::string>& contents ) {
  LogTrace(mlDqmClient_)
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
	 idir->find("EvF") == std::string::npos ) { continue; }
    
    // Extract source directory path 
    std::string source_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey path( source_dir );
    
    // Check path is valid to level of a module
    if ( path.granularity() != sistrip::CCU_CHAN ) { continue; } //@@ 
    
    // Generate corresponding client path (removing trailing "/")
    std::string client_dir = path.path();
    client_dir = client_dir.substr( 0, client_dir.size()-1 ); 
    
    // Iterate through MonitorElements from pwd directory
    mui_->setCurrentFolder( source_dir );
    std::vector<std::string> me_list = mui_->getMEs();
    std::vector<std::string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve granularity from histogram title (necessary?)
      SiStripHistoTitle title( *ime );
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = SiStripFecKey::lldChan(title.channel());
      } else if ( title.granularity() == sistrip::LLD_CHAN ) {
	channel = title.channel();
      } else {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " Unexpected histogram granularity: "
	  << title.granularity();
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
	while ( ihis < ihistos->second.end() ) {
	  if ( (*ime) == (*ihis)->title_ ) { break; }
	  ihis++;
	}
      }
      
      // Insert CollateME into map if missing
      if ( !histo ) {

	// Retrieve ME pointer
	MonitorElement* me = mui_->get( mui_->pwd()+"/"+(*ime) );
	
	// Check if profile or 1D
	TProfile* prof = ExtractTObject<TProfile>().extract( me );
	TH1F* his = ExtractTObject<TH1F>().extract( me );

	// Create CME and extract ME*
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
	    histo->me_ = histo->cme_->getMonitorElement(); 
	    mui_->add( histo->cme_, mui_->pwd()+"/"+(*ime) );
	  }
	}
	
      }

      // Add to CollateME
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
  
  printHistosMap();
  
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
  cout << endl // LogTrace(mlDqmClient_)
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



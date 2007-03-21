#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui,
						  const sistrip::RunType& task ) 
  : mui_(mui),
    collations_(),
    action_(sistrip::NO_ACTION),
    task_(task)
{
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Constructing object...";
  collations_.clear();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const std::vector<std::string>& contents ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Creating collated histograms...";
  
  if ( contents.empty() ) { return; }
  
  std::vector<std::string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Ignore directories on client side
    if ( idir->find("Collector") == std::string::npos ) { continue; }
    
    // Extract directory paths
    std::string collector_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey path( collector_dir );
    if ( path.fecCrate() == sistrip::invalid_ ||
	 path.fecSlot() == sistrip::invalid_ ||
	 path.fecRing() == sistrip::invalid_ ||
	 path.ccuAddr() == sistrip::invalid_ ||
	 path.ccuChan() == sistrip::invalid_ ) { continue; } 
    
    std::string dir = path.path();
    std::string client_dir = dir.substr( 0, dir.size()-1 ); 

    // Retrieve MonitorElements from pwd directory
    mui()->setCurrentFolder( collector_dir );
    std::vector<std::string> me_list = mui()->getMEs();

    // Iterate through MEs and create CMEs
    std::vector<std::string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve granularity from histogram title (necessary?)
      SiStripHistoTitle title( *ime );
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = (title.channel()-32)/2;
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

      // Fill FED-FEC std::map
      mapping_[title.keyValue()] = fec_key;

      // Find CollateME in collations std::map
      CollateMonitorElement* cme = 0;
      CollationsMap::iterator ikey = collations_.find( fec_key );
      if ( ikey != collations_.end() ) { 
	Collations::iterator ihis = ikey->second.begin();
	while ( !cme && ihis != ikey->second.end() ) {
	  if ( (*ime) == ihis->first ) { 
	    SiStripFecKey path(ikey->first);
	    std::string dir = path.path();
	    cme = ihis->second; 
	  }
	  ihis++;
	}
      } else { cme = 0; }
	
      // Create CollateME if it doesn't exist
      if ( !cme ) {
	  
	// Retrieve ME pointer
	MonitorElement* me = mui()->get( mui()->pwd()+"/"+(*ime) );

	// Create profile CME
	TProfile* prof = ExtractTObject<TProfile>().extract( me );
	if ( prof ) { 
	  cme = mui()->collateProf( (*ime), (*ime), client_dir ); 
	  if ( cme ) { 
	    mui()->add( cme, mui()->pwd()+"/"+(*ime) );
	    collations_[fec_key].push_back( Collation((*ime),cme) );
	  }
	}

	// Create one-dim CME
	TH1F* his = ExtractTObject<TH1F>().extract( me );
	if ( prof ) { prof->SetErrorOption("s"); } //@@ necessary?
	else if ( his ) { 
	  cme = mui()->collate1D( (*ime), (*ime), client_dir ); 
	  if ( cme ) { 
	    mui()->add( cme, mui()->pwd()+"/"+(*ime) ); 
	    collations_[fec_key].push_back( Collation((*ime),cme) );
	  }
	}

      }
	  
      // Add to CME if found in collations std::map
      CollationsMap::iterator jkey = collations_.find( fec_key );
      if ( jkey != collations_.end() ) { 
	Collations::iterator ihis = jkey->second.begin();
	while ( ihis != jkey->second.end() ) {
	  if ( (*ime) == ihis->first ) { 
	    if ( ihis->second ) {
	      mui()->add( ihis->second, mui()->pwd()+"/"+(*ime) );
	    }
	  }
	  ihis++;
	} 
	  
      }
 
    }
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
  std::string pwd = mui()->pwd();
  mui()->setCurrentFolder( directory );

  // Book new histogram
  std::string name = SummaryGenerator::name( task_, mon, pres, view, directory );
  MonitorElement* me = mui()->get( mui()->pwd() + "/" + name );
  if ( !me ) { 
    if ( pres == sistrip::SUMMARY_HISTO ) { 
      me = mui()->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( pres == sistrip::SUMMARY_1D ) { 
      me = mui()->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( pres == sistrip::SUMMARY_2D ) { 
      me = mui()->getBEInterface()->book2D( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else if ( pres == sistrip::SUMMARY_PROF ) { 
      me = mui()->getBEInterface()->bookProfile( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else { me = 0; 
    }
  }
  TH1F* summary = ExtractTObject<TH1F>().extract( me ); 
  
  // Return to pwd
  mui()->setCurrentFolder( pwd );
  
  return summary;
  
}


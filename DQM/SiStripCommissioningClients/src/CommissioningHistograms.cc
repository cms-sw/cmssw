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
						  const sistrip::Task& task ) 
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
void CommissioningHistograms::createCollations( const vector<string>& contents ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Creating collated histograms...";
  
  if ( contents.empty() ) { return; }
  
  vector<string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Ignore directories on client side
    if ( idir->find("Collector") == string::npos ) { continue; }
    
    // Extract directory paths
    string collector_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey::Path path = SiStripHistoNamingScheme::controlPath( collector_dir );
    if ( path.fecCrate_ == sistrip::invalid_ ||
	 path.fecSlot_ == sistrip::invalid_ ||
	 path.fecRing_ == sistrip::invalid_ ||
	 path.ccuAddr_ == sistrip::invalid_ ||
	 path.ccuChan_ == sistrip::invalid_ ) { continue; } 
    
    string dir = SiStripHistoNamingScheme::controlPath( path );
    string client_dir = dir.substr( 0, dir.size()-1 ); 

    // Retrieve MonitorElements from pwd directory
    mui()->setCurrentFolder( collector_dir );
    vector<string> me_list = mui()->getMEs();

    // Iterate through MEs and create CMEs
    vector<string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve granularity from histogram title (necessary?)
      HistoTitle title = SiStripHistoNamingScheme::histoTitle( *ime );
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity_ == sistrip::APV ) {
	channel = (title.channel_-32)/2;
      } else if ( title.granularity_ == sistrip::LLD_CHAN ) {
	channel = title.channel_;
      } else {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " Unexpected histogram granularity: "
	  << title.granularity_;
      }

      // Build FEC key
      uint32_t fec_key = SiStripFecKey::key( path.fecCrate_,
					     path.fecSlot_,
					     path.fecRing_,
					     path.ccuAddr_,
					     path.ccuChan_,
					     channel );

      // Fill FED-FEC map
      mapping_[title.keyValue_] = fec_key;

      // Find CollateME in collations map
      CollateMonitorElement* cme = 0;
      CollationsMap::iterator ikey = collations_.find( fec_key );
      if ( ikey != collations_.end() ) { 
	Collations::iterator ihis = ikey->second.begin();
	while ( !cme && ihis != ikey->second.end() ) {
	  if ( (*ime) == ihis->first ) { 
	    SiStripFecKey::Path path = SiStripFecKey::path(ikey->first);
	    string dir = SiStripHistoNamingScheme::controlPath(path);
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
	  
      // Add to CME if found in collations map
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
						  const string& directory,
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
						  pair<string,
						  sistrip::Granularity> summ1 ) {
  createSummaryHisto( summ0.first, summ0.second, summ1.first, summ1.second );
}

// -----------------------------------------------------------------------------
// 
TH1* CommissioningHistograms::histogram( const sistrip::Monitorable& mon, 
					 const sistrip::Presentation& pres, 
					 const sistrip::View& view,
					 const string& directory,
					 const uint32_t& xbins ) {
  
  // Remember pwd 
  string pwd = mui()->pwd();
  mui()->setCurrentFolder( directory );

  // Book new histogram
  string name = SummaryGenerator::name( task_, mon, pres, view, directory );
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


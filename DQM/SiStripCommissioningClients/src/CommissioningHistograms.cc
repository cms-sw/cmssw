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
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " Constructing object...";
  collations_.clear();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const vector<string>& contents ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " Creating CollateMonitorElements...";

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
    //cout << "DIR: " << dir << " " << client_dir << endl;

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
	cerr << endl // edm::LogWarning(mlDqmClient_)
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

      if ( 1 ) { 

	// Find CollateME in collations map
	CollateMonitorElement* cme = 0;
	CollationsMap::iterator ikey = collations_.find( fec_key );
	if ( ikey != collations_.end() ) { 
	  Collations::iterator ihis = ikey->second.begin();
	  while ( !cme && ihis != ikey->second.end() ) {
	    if ( (*ime) == ihis->first ) { 
	      SiStripFecKey::Path path = SiStripFecKey::path(ikey->first);
	      string dir = SiStripHistoNamingScheme::controlPath(path);
	      //cout << "CME EXISTS: " << dir+ihis->first << " ptr: " << ihis->second << endl;
	      cme = ihis->second; 
	    }
	    ihis++;
	  }
	} else { cme = 0; }
	
	// Create CollateME if it doesn't exist
	if ( !cme ) {
	  
	  // Retrieve ME pointer
	  MonitorElement* me = mui()->get( mui()->pwd()+"/"+(*ime) );
	  //cout << "SOURCE ME: " << mui()->pwd()+"/"+(*ime) << endl;

	  // Create profile CME
	  TProfile* prof = ExtractTObject<TProfile>().extract( me );
	  if ( prof ) { 
	    cme = mui()->collateProf( (*ime), (*ime), client_dir ); 
	    if ( cme ) { 
	      mui()->add( cme, mui()->pwd()+"/"+(*ime) );
	      collations_[fec_key].push_back( Collation((*ime),cme) );
	      //MonitorElement* me = cme->getMonitorElement();
	      //if ( me ) { cout << "CREATED NEW PROF CME: " << client_dir << "/" << me->getName() << endl; }
	      //MonitorElement* me1 = mui()->get( client_dir+"/"+(*ime) );
	      //if ( me1 ) { cout << "FOUND NEW PROF CME: " << client_dir << "/" << me1->getName() << endl; }
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
	      //MonitorElement* me = cme->getMonitorElement();
	      //if ( me ) { cout << "CREATED NEW 1D CME: " << client_dir << "/" << me->getName() << endl; }
	      //MonitorElement* me1 = mui()->get( client_dir+"/"+(*ime) );
	      //if ( me1 ) { cout << "FOUND NEW PROF CME: " << client_dir << "/" << me1->getName() << endl; }
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
		//cout << "ADDED TO CME: " << mui()->pwd() << "/" << (*ime) << endl;
	      }
	    }
	    ihis++;
	  } 
	  
	}

      } else { // 
	
	// Find CollateME in collations map
	CollateMonitorElement* cme = 0;
	CollationsMap::iterator ikey = collations_.find( fec_key );
	if ( ikey != collations_.end() ) { 
	  Collations::iterator ihis = ikey->second.begin();
	  while ( !cme && ihis != ikey->second.end() ) {
	    if ( "Collated_"+(*ime) == ihis->first ) { 
	      SiStripFecKey::Path path = SiStripFecKey::path(ikey->first);
	      string dir = SiStripHistoNamingScheme::controlPath(path);
	      cout << "CME EXISTS: " << dir+ihis->first << " ptr: " << ihis->second << endl;
	      cme = ihis->second; 
	    }
	    ihis++;
	  }
	} else { cme = 0; }
      
	// Create CollateME if it doesn't exist
	if ( !cme ) {
	  cout << "SOURCE ME: " << mui()->pwd()+"/"+(*ime) << endl;
	  MonitorElement* me = mui()->get( mui()->pwd()+"/"+(*ime) );
	  TProfile* prof = ExtractTObject<TProfile>().extract( me );
	  TH1F* his = ExtractTObject<TH1F>().extract( me );
	  if ( prof ) { prof->SetErrorOption("s"); } //@@ necessary?
	  if ( prof ) { cme = mui()->collateProf( "Collated_"+(*ime), "Collated_"+(*ime), client_dir ); }
	  else if ( his ) { cme = mui()->collate1D( "Collated_"+(*ime), "Collated_"+(*ime), client_dir ); }
	  else { 
	    cout << "PROBLEM..." << endl;
	    cme = 0; 
	  }
	  // Record name and pointer in map
	  if ( cme ) { 
	    if ( cme->getMonitorElement() ) {
	      cout << "CREATED NEW CME: " << client_dir << "/" << cme->getMonitorElement()->getName() << endl;
	    } else {
	      cout << "NULL PTR TO ME FROM CME: " << client_dir << endl;
	    }
	    collations_[fec_key].push_back( Collation("Collated_"+(*ime),cme) );
	    if ( collations_[fec_key].capacity() < 10 ) { collations_[fec_key].reserve(10); }
	    MonitorElement* me = mui()->get( client_dir+"/"+"Collated_"+(*ime) );
	    if ( me ) {
	      cout << "NEW CME LOCATION: " << client_dir << "/" << me->getName() << endl;
	    }
	  }
	}

	// Find CollateME in collations map (again)
	cme = 0;
	CollationsMap::iterator jkey = collations_.find( fec_key );
	if ( jkey != collations_.end() ) { 
	  Collations::iterator ihis = jkey->second.begin();
	  while ( !cme && ihis != jkey->second.end() ) {
	    if ( "Collated_"+(*ime) == ihis->first ) { cme = ihis->second; }
	    ihis++;
	  }
	} else { cme = 0; }
      
	// "Add" to CollateME if it exists
	if ( cme ) {
	  cout << "ADDED TO CME: " << collector_dir << "/" << (*ime) << endl;
	  mui()->add( cme, collector_dir+"/"+(*ime) ); // note search pattern
	} 
     
      } //
 
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
void CommissioningHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
						  const sistrip::SummaryType& type, 
						  const string& directory,
						  const sistrip::Granularity& gran ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " (Derived) implementation to come..."; 
}

// -----------------------------------------------------------------------------
/** Wraps other createSummaryHisto() method. */
void CommissioningHistograms::createSummaryHisto( pair<sistrip::SummaryHisto,
						  sistrip::SummaryType> summ0, 
						  pair<string,
						  sistrip::Granularity> summ1 ) {
  createSummaryHisto( summ0.first, summ0.second, summ1.first, summ1.second );
}

// -----------------------------------------------------------------------------
// 
TH1* CommissioningHistograms::histogram( const sistrip::SummaryHisto& histo, 
					 const sistrip::SummaryType& type, 
					 const sistrip::View& view,
					 const string& directory,
					 const uint32_t& xbins ) {
  
  string name = SummaryGenerator::name( task_, histo, type, view, directory );
  mui()->setCurrentFolder( directory );
  MonitorElement* me = mui()->get( mui()->pwd() + "/" + name );
  if ( !me ) { 
    if ( type == sistrip::SUMMARY_DISTR ) { 
      me = mui()->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( type == sistrip::SUMMARY_1D ) { 
      me = mui()->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( type == sistrip::SUMMARY_2D ) { 
      me = mui()->getBEInterface()->book2D( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else if ( type == sistrip::SUMMARY_PROF ) { 
      me = mui()->getBEInterface()->bookProfile( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else { me = 0; 
    }
  }
  TH1F* summary = ExtractTObject<TH1F>().extract( me ); 
  return summary;
}


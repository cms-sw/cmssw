#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include <sstream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui ) 
  : mui_(mui),
    collations_(),
    action_(sistrip::NO_ACTION)
{
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Created base object!" << endl;
  collations_.clear();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const vector<string>& contents ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  if ( contents.empty() ) { return; }
  
  vector<string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Extract directory paths
    string collector_dir = idir->substr( 0, idir->find(":") );
    //cout << "Collector dir: " << collector_dir << endl;
    const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( collector_dir );
    //     cout << "ControlPath: " 
    // 	 << path.fecCrate_ << "/" 
    // 	 << path.fecSlot_ << "/" 
    // 	 << path.fecRing_ << "/" 
    // 	 << path.ccuAddr_ << "/" 
    // 	 << path.ccuChan_ << endl;
    string client_dir = SiStripHistoNamingScheme::controlPath( path.fecCrate_,
							       path.fecSlot_,
							       path.fecRing_,
							       path.ccuAddr_,
							       path.ccuChan_ );
    //cout << "Client dir: " << collector_dir << endl;
    
    if ( path.fecCrate_ == sistrip::invalid_ ||
	 path.fecSlot_ == sistrip::invalid_ ||
	 path.fecRing_ == sistrip::invalid_ ||
	 path.ccuAddr_ == sistrip::invalid_ ||
	 path.ccuChan_ == sistrip::invalid_ ) { continue; } 

    // Retrieve MonitorElements from pwd directory
    mui()->setCurrentFolder( collector_dir );
    vector<string> me_list = mui()->getMEs();
    //cout << "Found " << me_list.size() << " MEs in Collector dir " << collector_dir << endl;
    
    CollateMonitorElement* cme = 0;
    vector<string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve pointer to monitor element
      MonitorElement* me = mui()->get( mui()->pwd()+"/"+(*ime) ); // path + name
      TProfile* prof = ExtractTObject<TProfile>().extract( me );
      TH1F* his = ExtractTObject<TH1F>().extract( me );
      if ( prof ) { prof->SetErrorOption("s"); } //@@ is this necessary? (until bug fix applied to dqm)...

      static SiStripHistoNamingScheme::HistoTitle title;
      title = SiStripHistoNamingScheme::histoTitle( *ime );
      uint32_t key = SiStripControlKey::key( sistrip::invalid_, //@@ WARNING: only good for one partition only!!!
					     path.fecSlot_,
					     path.fecRing_,
					     path.ccuAddr_,
					     path.ccuChan_,
					     title.channel_ );
      if ( title.granularity_ != sistrip::LLD_CHAN ) {
	cerr << "[" << __PRETTY_FUNCTION__ << "] Unexpected histogram granularity: " << title.granularity_ << endl;
      }
      
      // Fill map linking FED key to FEC key
      mapping_[title.keyValue_] = key;

      // Create collation MEs
      CollationsMap::iterator iter = collations_.find( key );
      if ( iter == collations_.end() ) {
	if ( prof )     { cme = mui()->collateProf( *ime, *ime, client_dir ); }
	else if ( his ) { cme = mui()->collate1D( *ime, *ime, client_dir ); }
	else { cme = 0; cerr << "[" << __PRETTY_FUNCTION__ << "] NULL pointers to histos!" << endl; }
	if ( cme ) {
	  mui()->add( cme, "*/"+client_dir+(*ime) ); // note search pattern
	  collations_[key].push_back( client_dir+(*ime) ); // store "path + name"
	}
      } else {
	if ( find( iter->second.begin(), iter->second.end(), client_dir+(*ime) ) == iter->second.end() ) {
	  if ( prof )     { cme = mui()->collateProf( *ime, *ime, client_dir ); }
	  else if ( his ) { cme = mui()->collate1D( *ime, *ime, client_dir ); }
	  else { cme = 0; cerr << "[" << __PRETTY_FUNCTION__ << "] NULL pointers to histos!" << endl; }
	  if ( cme ) {
	    mui()->add( cme, "*/"+client_dir+(*ime) ); // note search pattern
	    collations_[key].push_back( client_dir+(*ime) ); // store "path + name"
	  }
	}
      }
    }
  }
  
}

// // -----------------------------------------------------------------------------
// /** */
// void CommissioningHistograms::subscribe( MonitorUserInterface* mui, string pattern ) {
//   cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
//   if ( mui ) { mui->subscribe(pattern); }
// }

// // -----------------------------------------------------------------------------
// /** */
// void CommissioningHistograms::unsubscribe( MonitorUserInterface* mui, string pattern ) {
//   cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
//   if ( mui ) { mui->unsubscribe(pattern); }
// }

// // -----------------------------------------------------------------------------
// /** */
// void CommissioningHistograms::saveHistos( MonitorUserInterface* mui, string name ) {
//   stringstream ss; 
//   if ( name == "" ) { ss << "Client.root"; }
//   else { ss << name; }
//   cout << "[" << __PRETTY_FUNCTION__ << "]" 
//        << " Saving histogams to file '" << ss.str() << "'..." << endl;
//   if ( mui ) { mui->save( ss.str() ); }
// }

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** Wraps other createSummaryHisto() method. */
void CommissioningHistograms::createSummaryHisto( pair<sistrip::SummaryHisto,
						  sistrip::SummaryType> summ, 
						  string directory ) {
  createSummaryHisto( summ.first, summ.second, directory );
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
						  const sistrip::SummaryType& type, 
						  const string& directory ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " (Derived) implementation to come..." << endl;
}


// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " (Derived) implementation to come..." << endl;
}

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
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::subscribeNew() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  if ( mui_ ) { mui_->subscribeNew("*"); }
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

      // Create collation MEs
      if ( find( collations_.begin(), collations_.end(), client_dir+(*ime) ) == collations_.end() ) {
	//cout << "Entry " << client_dir+(*ime) << " is not found in 'collations' vector" << endl;
	if ( prof )     { cme = mui()->collateProf( *ime, *ime, client_dir ); }
	else if ( his ) { cme = mui()->collate1D( *ime, *ime, client_dir ); }
	else            { 
	  cme = 0; 
	  cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	       << " NULL pointers to histos!" << endl; 
	}
	if ( cme ) {
	  mui()->add( cme, "*/"+client_dir+(*ime) ); // note search pattern
	  collations_.push_back( client_dir+(*ime) ); // record "path + name"
	  //cout << "Created collate ME with name " << *ime 
	  //<< " in directory " << client_dir 
	  //<< " which collates all histos matching string: " 
	  //<< ( "*/"+client_dir+(*ime) ) << endl;
	  //cout << "Number of collate MEs is " << collations_.size() << endl;
	} 
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::saveHistos( string name ) {
  stringstream ss; 
  if ( name == "" ) { ss << "Client.root"; }
  else { ss << name; }
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " Saving histogams to file '" << ss.str() << "'..." << endl;
  if ( mui_ ) { mui_->save( ss.str() ); }
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
void CommissioningHistograms::createTrackerMap() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" 
       << " (Derived) implementation to come..." << endl;
}

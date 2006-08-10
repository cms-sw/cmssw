#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"

using namespace std;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui ) 
  : mui_(mui),
    collations_()
    //factory_( new SummaryHistogramFactory<CommissioningAnalysis::Monitorables>() )
{
  cout << "[CommissioningHistograms::CommissioningHistograms]" 
       << " Created base object!" << endl;
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << "[CommissioningHistograms::~CommissioningHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const vector<string>& added_contents ) { //@@ what about removed contents?
  static const string method = "CommissioningHistograms::createCollations";

  if ( added_contents.empty() ) { return; }
  
  vector<string>::const_iterator idir;
  for ( idir = added_contents.begin(); idir != added_contents.end(); idir++ ) {
    
    // Extract directory paths
    string collector_dir = idir->substr( 0, idir->find(":") );
    const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( collector_dir );
    string client_dir = SiStripHistoNamingScheme::controlPath( path.fecCrate_,
							       path.fecSlot_,
							       path.fecRing_,
							       path.ccuAddr_,
							       path.ccuChan_ );
    
    if ( path.fecCrate_ == sistrip::all_ ||
	 path.fecSlot_ == sistrip::all_ ||
	 path.fecRing_ == sistrip::all_ ||
	 path.ccuAddr_ == sistrip::all_ ||
	 path.ccuChan_ == sistrip::all_ ) { continue; } 
    
    // Retrieve MonitorElements from pwd directory
    mui()->setCurrentFolder( collector_dir );
    vector<string> me_list = mui()->getMEs();
    
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
	if ( prof )     { cme = mui()->collateProf( *ime, *ime, client_dir ); }
	else if ( his ) { cme = mui()->collate1D( *ime, *ime, client_dir ); }
	else            { cme = 0; cerr << "["<<method<<"]" << "NULL pointers to histos!" << endl; }
	if ( cme ) {
	  mui()->add( cme, "*/"+client_dir+(*ime) ); // note search pattern
	  collations_.push_back( client_dir+(*ime) ); // record "path + name"
	} 
      }
      
    }
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis() {
  cout << "[CommissioningHistograms::histoAnalysis]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHistos( const vector<sistrip::SummaryHisto>& histos, 
						   const sistrip::SummaryType& type, 
						   const string& directory ) {
  cout << "[CommissioningHistograms::createSummaryHistos]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createTrackerMap() {
  cout << "[CommissioningHistograms::createTrackerMap]" 
       << " (Derived) implementation to come..." << endl;
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << "[CommissioningHistograms::uploadToConfigDb]" 
       << " (Derived) implementation to come..." << endl;
}

// // -----------------------------------------------------------------------------
// /** */
// void CommissioningHistograms::bookDqmHisto( vector<TH1F*>& dqm_histos ) {
//   static const string method = "CommissioningHistograms::bookDqmHistos";
  
//   // Book 1D histo using DQM fwk 
//   MonitorElement* me = mui()->getBEInterface()->book1D( "name", "title", 1, 0., 1. );
  
//   // Extract TH1F
//   TH1F* his = ExtractTObject<TH1F>().extract( me );
//   if ( his ) { 
//     dqm_histos.push_back( his ); 
//   } else {
//     cerr << "["<<method<<"] NULL pointer to TH1F!" << endl; 
//     return; 
//   }
  
// }



#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"

// This line is necessary
XDAQ_INSTANTIATOR_IMPL(SiStripCommissioningClient);

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningClient::SiStripCommissioningClient( xdaq::ApplicationStub* stub ) 
  : DQMBaseClient( stub, "SiStripCommissioningClient", "localhost", 9090 ),
    web_(0),
    histos_(0),
    collations_()
{
  web_ = new SiStripCommissioningWebClient( this,
					    this->getContextURL(),
					    this->getApplicationURL(), 
					    &(this->mui_) );
  xgi::bind( this, &SiStripCommissioningClient::handleWebRequest, "Request" );
}

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningClient::~SiStripCommissioningClient() {
  if ( web_ ) { delete web_; }
  if ( histos_ ) { delete histos_; }
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Configured" state. */
void SiStripCommissioningClient::configure() {
  collations_.clear();
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Enabled" state. */
void SiStripCommissioningClient::newRun() {
  ( this->upd_ )->registerObserver( this ); 
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Halted" state. */
void SiStripCommissioningClient::endRun() {
  if ( mui_ ) { mui_->save("client.root"); }
  if ( histos_ ) { delete histos_; histos_ = 0; }
}

// -----------------------------------------------------------------------------
/** Called by the "Updater" following each update. */
void SiStripCommissioningClient::onUpdate() const {

  // Subscribe to new monitorables and retrieve updated contents
  ( this->mui_ )->subscribeNew( "*" );
  vector<string> added_contents;
  ( this->mui_ )->getAddedContents( added_contents );
  if ( added_contents.empty() ) { return; }
  
  // Extract commissioning task from added contents
  sistrip::Task task = extractTask( added_contents );

  // Create histograms for given commissioning task
  createTaskHistograms( task );

  // Create collation histograms based on added contents
  createCollations( added_contents );
  
}


// -----------------------------------------------------------------------------
/** Extract "commissioning task" string from "added contents". */
sistrip::Task SiStripCommissioningClient::extractTask( const vector<string>& added_contents ) const {
  sistrip::Task task = sistrip::NO_TASK;

  // Iterate through added contents
  vector<string>::const_iterator istr = added_contents.begin();
  while ( istr != added_contents.end() && 
	  task != sistrip::NO_TASK ) {
    // Search for "commissioning task" string
    string::size_type pos = istr->find( sistrip::commissioningTask_ );
    if ( pos != string::npos ) { 
      // Extract commissioning task from string 
      string str = istr->substr( pos+sistrip::commissioningTask_.size()+1, string::npos ); 
      if ( !str.empty() ) { 
	task = SiStripHistoNamingScheme::task( str ); 
	cout << "[SiStripCommissioningClient::onUpdate]"
	     << " Found 'SiStripCommissioningTask' string with value " 
	     << str << endl;
      }
    }
    istr++;
  }
  return task;
}

// -----------------------------------------------------------------------------
/** Create histograms for given commissioning task. */
void SiStripCommissioningClient::createTaskHistograms( const sistrip::Task& task ) const {

  // Check if object already exists
  if ( histos_ ) { return; }

  // Create corresponding "commissioning histograms" object 
  if      ( task == sistrip::APV_TIMING )  { histos_ = new ApvTimingHistograms( mui_ ); }
  else if ( task == sistrip::FED_CABLING ) { histos_ = new FedCablingHistograms( mui_ ); }
  else if ( task == sistrip::FED_TIMING )  { histos_ = new FedTimingHistograms( mui_ ); }
  else if ( task == sistrip::OPTO_SCAN )   { histos_ = new OptoScanHistograms( mui_ ); }
  else if ( task == sistrip::PEDESTALS )   { histos_ = new PedestalsHistograms( mui_ ); }
  else if ( task == sistrip::VPSP_SCAN )   { histos_ = new VpspScanHistograms( mui_ ); }
  else if ( task == sistrip::NO_TASK ) { histos_ = 0; }
  else if ( task == sistrip::UNKNOWN_TASK ) {
    histos_ = 0;
    cerr << "[SiStripCommissioningClient::onUpdate]"
	 << " Unknown commissioning task!" << endl;
  }

}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::createCollations( const vector<string>& added_contents ) const {
  
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
    mui_->setCurrentFolder( collector_dir );
    vector<string> me_list = mui_->getMEs();
    
    uint16_t n_cme = 0;
    vector<string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {

      string cme_name = *ime;
      string cme_title = *ime;
      string cme_dir = client_dir;
      string search_str = "*/" + client_dir + *ime;

      // Retrieve pointer to monitor element
      string path_and_title = this->mui_->pwd() + "/" + *ime;
      MonitorElement* me = this->mui_->get( path_and_title );
      TProfile* prof = ExtractTObject<TProfile>()( me );
      TH1F* his = ExtractTObject<TH1F>()( me );
      if ( prof ) { prof->SetErrorOption("s"); } //@@ necessary until bug fix applied to dqm...
      
      if ( find( collations_.begin(), collations_.end(), search_str ) == collations_.end() ) {
	// Collate TProfile histos
	if ( prof ) { 
	  CollateMonitorElement* cme = mui_->collateProf( cme_name, cme_title, cme_dir );
	  mui_->add( cme, search_str );
	  collations_.push_back( search_str );
	  n_cme++;
	} 
	// Collate TH1F histos
	if ( his ) { 
	  CollateMonitorElement* cme = mui_->collate1D( cme_name, cme_title, cme_dir );
	  mui_->add( cme, search_str );
	  collations_.push_back( search_str );
	  n_cme++;
	} 
      }

    }

  }

}

// -----------------------------------------------------------------------------
/** General access to client info. */
void SiStripCommissioningClient::general( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ) {
  if ( web_ ) { web_->Default( in, out ); }
  else { cerr << "[SiStripCommissioningClient::general]"
	      << "Null pointer for web interface!" << endl; }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::handleWebRequest( xgi::Input* in, xgi::Output* out ) {
  if ( web_ ) { web_->handleRequest(in, out); }
  else {
    cerr << "[SiStripCommissioningClient::general]"
	 << "Null pointer for web interface!" << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms* histos( const SiStripCommissioningClient& client ) { 
  if ( !client.histos_ ) {
    cerr << "[SiStripCommissioningClient::histos]"
	 << " Null pointer to CommissioningHistograms object!" << endl; 
  }
  return client.histos_;
}

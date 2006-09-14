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
    histos_(0)
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
void SiStripCommissioningClient::configure() {;}

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
  if ( histos_ ) { histos_->createCollations( added_contents ); }
  
  // Update monitorables using histogram analysis
  if ( histos_ ) { histos_->histoAnalysis(); }
  
}

// -----------------------------------------------------------------------------
/** Extract "commissioning task" string from "added contents". */
sistrip::Task SiStripCommissioningClient::extractTask( const vector<string>& added_contents ) const {
  sistrip::Task task = sistrip::UNDEFINED_TASK;

  // Iterate through added contents
  vector<string>::const_iterator istr = added_contents.begin();
  while ( istr != added_contents.end() && 
	  task != sistrip::UNDEFINED_TASK ) {
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
  if      ( task == sistrip::APV_TIMING )     { histos_ = new ApvTimingHistograms( mui_ ); }
  else if ( task == sistrip::FED_CABLING )    { histos_ = new FedCablingHistograms( mui_ ); }
  else if ( task == sistrip::FED_TIMING )     { histos_ = new FedTimingHistograms( mui_ ); }
  else if ( task == sistrip::OPTO_SCAN )      { histos_ = new OptoScanHistograms( mui_ ); }
  else if ( task == sistrip::PEDESTALS )      { histos_ = new PedestalsHistograms( mui_ ); }
  else if ( task == sistrip::VPSP_SCAN )      { histos_ = new VpspScanHistograms( mui_ ); }
  else if ( task == sistrip::UNDEFINED_TASK ) { histos_ = 0; }
  else if ( task == sistrip::UNKNOWN_TASK ) {
    histos_ = 0;
    cerr << "[SiStripCommissioningClient::onUpdate]"
	 << " Unknown commissioning task!" << endl;
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

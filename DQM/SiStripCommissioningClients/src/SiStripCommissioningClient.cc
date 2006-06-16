#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"

// This line is necessary
XDAQ_INSTANTIATOR_IMPL(SiStripCommissioningClient);

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningClient::SiStripCommissioningClient( xdaq::ApplicationStub* stub ) 
  : DQMBaseClient( stub, "SiStripCommissioningClient", "localhost", 9090 ),
    web_(0),
    histos_(0),
    initialNumOfUpdates_(0) 
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
void SiStripCommissioningClient::configure() {}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Enabled" state. */
void SiStripCommissioningClient::newRun() {
  ( this->upd_ )->registerObserver( this ); // Register with the Updater object
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Halted" state. */
void SiStripCommissioningClient::endRun() {
  // Do histogram analysis and upload monitorables to database
  if ( histos_ ) { 
    // histos_->createProfileHistos(); 
    // histos_->histoAnalysis(); 
    // histos_->createSummaryHistos(); 
    // histos_->uploadToConfigDb(); 
    // histos_->saveToFile(); 
  }
  if ( mui_ ) { mui_->save("client.root"); }
}

// -----------------------------------------------------------------------------
/** Called by the "Updater" following each update. */
void SiStripCommissioningClient::onUpdate() const {

  // Number of updates received by collector
  uint32_t num_of_updates = mui_->getNumUpdates();
  if ( !initialNumOfUpdates_ ) { initialNumOfUpdates_ = num_of_updates; }
  cout << "[SiStripCommissioningClient::onUpdate]"
       << " Number of updates: " << num_of_updates - initialNumOfUpdates_ << endl; 

  // Subscribe to new monitorables and retrieve updated contents
  ( this->mui_ )->subscribeNew( "*" ); //@@ temporary?
  vector<string> added_contents;
  ( this->mui_ )->getAddedContents( added_contents );
  if ( added_contents.empty() ) { 
    // cout << "[SiStripCommissioningClient::onUpdate] No added contents!" << endl;
    return; 
  }
  // cout << "[SiStripCommissioningClient::onUpdate] Number of added contents is: " << added_contents.size() << endl;
  
  // Create CommissioningHistogram object 
  createCommissioningHistos( added_contents );
  
  // Create Collation histos based on added contents
  if ( histos_ ) { 
    histos_->createCollateMEs( added_contents ); 
    histos_->createProfileHistos(); 
    // histos_->histoAnalysis(); 
    // histos_->createSummaryHistos(); 
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

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::createCommissioningHistos( const vector<string>& added_contents ) const {
  cout << "[SiStripCommissioningClient::createCommissioningHistos]" << endl;
  
  if ( added_contents.empty() ) { return; }
  if ( histos_ ) { return; }
  
  // Iterate through "added contents" and retrieve "commissioning task" 
  string task_str = "";
  sistrip::Task task = sistrip::NO_TASK;
  vector<string>::const_iterator istr;
  for ( istr = added_contents.begin(); istr != added_contents.end(); istr++ ) { 
    string pattern = sistrip::commissioningTask_;
    string::size_type pos = istr->find( pattern );
    if ( pos != string::npos ) { 
      task_str = istr->substr( pos+pattern.size()+1, string::npos ); 
      if ( !task_str.empty() ) { task = SiStripHistoNamingScheme::task( task_str ); }
      continue; 
    }
  }
  cout << "[SiStripCommissioningClient::createCommissioningHistos]"
       << " Found 'SiStripCommissioningTask' string with value " 
       << task_str << endl;
  
  // Create corresponding "commissioning histograms" object 
  if      ( task == sistrip::APV_TIMING )  { histos_ = new ApvTimingHistograms( mui_ ); }
  else if ( task == sistrip::FED_CABLING ) { histos_ = new FedCablingHistograms( mui_ ); }
  else if ( task == sistrip::FED_TIMING )  { histos_ = new FedTimingHistograms( mui_ ); }
  else if ( task == sistrip::OPTO_SCAN )   { histos_ = new OptoScanHistograms( mui_ ); }
  else if ( task == sistrip::PEDESTALS )   { histos_ = new PedestalsHistograms( mui_ ); }
  else if ( task == sistrip::VPSP_SCAN )   { histos_ = new VpspScanHistograms( mui_ ); }
  else if ( task == sistrip::UNKNOWN_TASK ) {
    histos_ = 0;
    cerr << "[SiStripCommissioningClient::createCommissioningHistos]"
	 << " Unknown commissioning task!" << endl;
  } else if ( task == sistrip::NO_TASK ) { 
    histos_ = 0;
    cerr << "[SiStripCommissioningClient::createCommissioningHistos]"
	 << " No commissioning task string found!" << endl;
  }
  
}

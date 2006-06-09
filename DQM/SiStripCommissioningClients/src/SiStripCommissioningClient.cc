#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
//#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
//#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
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
    histo_(0) {
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
  if ( histo_ ) { delete histo_; }
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Configured" state. */
void SiStripCommissioningClient::configure() {
//   if ( mui_ ) { mui_->setVerbose(0); } 
//   else { //edm::LogError("SiStrip|Commissioning|Client") 
//     cerr << "[SiStripCommissioningClient::configure]"
// 	 << "Null pointer for MonitorUserInterface!" << endl;
//   }
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Enabled" state. */
void SiStripCommissioningClient::newRun() {
  // Register this "UpdateObserver" with the Updater object
  ( this->upd_ )->registerObserver( this );
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Halted" state. */
void SiStripCommissioningClient::endRun() {
  // Do histogram analysis and upload monitorables to database
  if ( histo_ ) { 
    histo_->createCollateMEs();
    histo_->createProfileHistos();
    histo_->createSummaryHistos();
    histo_->uploadToConfigDb(); 
  }
  if ( mui_ ) { mui_->save("SiStripCommissioningClient.root"); }
  else { //edm::LogError("SiStrip|Commissioning|Client") 
    cerr << "[SiStripCommissioningClient::endRun]"
	 << "Null pointer for MonitorUserInterface!" << endl;
  }
  
}

// -----------------------------------------------------------------------------
/** Called by the "Updater" following each update. */
void SiStripCommissioningClient::onUpdate() const {
  cout << "[SiStripCommissioningClient::onUpdate]"
       << "Number of updates: " << mui_->getNumUpdates() << endl;
  
  // Subscribe to new monitorables and retrieve updated contents
  ( this->mui_ )->subscribeNew( "*" ); //@@ temporary?
  vector<string> added_contents;
  ( this->mui_ )->getAddedContents( added_contents );
  if ( added_contents.empty() ) { return; }
  
  // Create CommissioningHistogram object 
  createCommissioningHistos( added_contents );
  
  //@@ anything here?
  if ( histo_ ) { 
    histo_->createCollateMEs();
    histo_->createProfileHistos();
  }
  
}

// -----------------------------------------------------------------------------
/** General access to client info. */
void SiStripCommissioningClient::general( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ) {
  if ( web_ ) { web_->Default( in, out ); }
  else { //edm::LogError("SiStrip|Commissioning|Client") 
    cerr << "[SiStripCommissioningClient::general]"
	 << "Null pointer for web interface!" << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::handleWebRequest( xgi::Input* in, xgi::Output* out ) {
  if ( web_ ) { web_->handleRequest(in, out); }
  else { //edm::LogError("SiStrip|Commissioning|Client") 
    cerr << "[SiStripCommissioningClient::general]"
	 << "Null pointer for web interface!" << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::createCommissioningHistos( const vector<string>& added_contents ) const {
  cout << "[SiStripCommissioningClient::createCommissioningHistos]" << endl;
  
  if ( added_contents.empty() ) { return; }
  if ( histo_ ) { return; }
  
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
  if      ( task == sistrip::APV_TIMING )  { histo_ = new ApvTimingHistograms( mui_ ); }
  //@@ else if ( task == sistrip::FED_CABLING ) { histo_ = new FedCablingHistograms( mui_ ); }
  else if ( task == sistrip::FED_TIMING )  { histo_ = new FedTimingHistograms( mui_ ); }
  else if ( task == sistrip::OPTO_SCAN )   { histo_ = new OptoScanHistograms( mui_ ); }
  else if ( task == sistrip::PEDESTALS )   { histo_ = new PedestalsHistograms( mui_ ); }
  else if ( task == sistrip::VPSP_SCAN )   { histo_ = new VpspScanHistograms( mui_ ); }
  else if ( task == sistrip::UNKNOWN_TASK ) {
    histo_ = 0;
    //edm::LogError("SiStrip|Commissioning|Client") 
    cerr << "[SiStripCommissioningClient::createCommissioningHistos]"
	 << "Unknown commissioning task!" << endl;
  } else if ( task == sistrip::NO_TASK ) { 
    cerr << "[SiStripCommissioningClient::createCommissioningHistos]"
	 << "No commissioning task string found!" << endl;
  }

}

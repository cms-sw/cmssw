#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/HistogramDisplayHandler.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include <SealBase/Callback.h>
#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/CgiUtils.h"
#include "cgicc/HTTPResponseHeader.h"
#include "cgicc/HTMLClasses.h"

// This line is necessary
XDAQ_INSTANTIATOR_IMPL(SiStripCommissioningClient);


using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningClient::SiStripCommissioningClient( xdaq::ApplicationStub* stub ) 
  : DQMBaseClient( stub, "SiStripCommissioningClient", "localhost", 9090 ),
    web_(0),
    histos_(0),
    task_(sistrip::UNKNOWN_TASK),
    first_(true)
{
  xgi::bind( this, &SiStripCommissioningClient::handleWebRequest, "Request" );
  xgi::bind( this, &SiStripCommissioningClient::CBHistogramViewer, "HistogramViewer" );
  fCallBack=new BSem(BSem::EMPTY);
  fCallBack->give();
  hdis_=NULL;
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
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]";
  web_ = new SiStripCommissioningWebClient( this,
					    getContextURL(),
					    getApplicationURL(), 
					    &mui_ );
  hdis_ = new HistogramDisplayHandler(mui_,fCallBack);
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Enabled" state. */
void SiStripCommissioningClient::newRun() {
  cout << endl // LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningClient::" << __func__ << "]";
  ( this->upd_ )->registerObserver( this ); 
  subscribeAll(); 
  first_ = true;
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Halted" state. */
void SiStripCommissioningClient::endRun() {
  cout << endl // LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningClient::" << __func__ << "]";
  unsubscribeAll(); 
  if ( histos_ ) { delete histos_; histos_ = 0; }
  if (hdis_) { delete hdis_; hdis_ = 0; }
}

// -----------------------------------------------------------------------------
/** Called by the "Updater" following each update. */
void SiStripCommissioningClient::onUpdate() const {
  
  if ( !mui_ ) {
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!";
    return;
  }
  
  // Print number of updates
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Number of updates: " << mui_->getNumUpdates();
  
  // Retrieve a list of all subscribed histograms
  //if ( first_ ) { mui_->subscribe( "*" ); first_ = false; }
  vector<string> contents;
  mui_->getContents( contents ); 
  
  if ( contents.empty() ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Request for 'Contents': vector is empty!";
    return; 
  }
  
  // Extract commissioning task from added contents
  if ( task_ == sistrip::UNKNOWN_TASK ) { task_ = extractTask( contents ); }
  
  // Create histograms for given commissioning task
  createHistograms( task_ );
  
  // Create collation histograms based on added contents
  if ( histos_ ) { histos_->createCollations( contents ); }
  
}

// -----------------------------------------------------------------------------
/** Extract "commissioning task" string from "added contents". */
sistrip::Task SiStripCommissioningClient::extractTask( const vector<string>& contents ) const {
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]";
  
  // Iterate through added contents
  vector<string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {
    
    // Search for "commissioning task" string
    string::size_type pos = istr->find( sistrip::commissioningTask_ );
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Looking for 'SiStripCommissioningTask' within string: " 
      << *istr;
    if ( pos != string::npos ) { 
      // Extract commissioning task from string 
      string value = istr->substr( pos+sistrip::commissioningTask_.size()+1, string::npos ); 
      if ( !value.empty() ) { 
	cout << endl // LogTrace(mlDqmClient_)
	  << "[SiStripCommissioningClient::" << __func__ << "]"
	  << " Found string " <<  istr->substr(pos,string::npos)
	  << " with value " << value;
	if ( !(mui_->get(sistrip::root_+"/"+istr->substr(pos,string::npos))) ) { 
	  mui_->setCurrentFolder(sistrip::root_);
	  mui_->getBEInterface()->bookString( istr->substr(pos,string::npos), value ); 
	}
	return SiStripHistoNamingScheme::task( value ); 
      }
    }
    istr++;
    
  }
  return sistrip::UNKNOWN_TASK;
}

// -----------------------------------------------------------------------------
/** Create histograms for given commissioning task. */
void SiStripCommissioningClient::createHistograms( const sistrip::Task& task ) const {

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
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Unknown commissioning task!";
  }
  
}

// -----------------------------------------------------------------------------
/** General access to client info. */
void SiStripCommissioningClient::general( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ) {
  if ( web_ ) { web_->Default( in, out ); }
  else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to WebPage!";
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::handleWebRequest( xgi::Input* in, xgi::Output* out ) {
  if ( web_ ) { web_->handleRequest(in, out); }
  else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to WebPage!"; 
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::CBHistogramViewer( xgi::Input* in, xgi::Output* out )  throw (xgi::exception::Exception) {

  if ( !hdis_ ) { return; }
  fCallBack ->take();

  seal::Callback action;
  action = seal::CreateCallback( hdis_, 
				 &HistogramDisplayHandler::HistogramViewer,
				 in, out ); 

  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
    fCallBack->take();
    fCallBack->give();
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    fCallBack->give();
    return;
  }

}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::histoAnalysis( bool debug ) {

  if ( !histos_ ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	 << " NULL pointer to CommissioningHistograms!"; 
    return;
  }
  
  seal::Callback action; 
  action = seal::CreateCallback( histos_, 
				 &CommissioningHistograms::histoAnalysis,
				 debug ); // no arguments
  
  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::subscribeAll( string pattern ) {

  if ( pattern == "" ) { pattern = "*/" + sistrip::root_ + "/*"; }

  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::subscribe,
				 pattern ); //@@ argument list

  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}
// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::updateHistos() {

  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::update
				 ); //@@ argument list
  
  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::unsubscribeAll( string pattern ) {
  
  if ( pattern == "" ) { pattern = "*/" + sistrip::root_ + "/*"; }
  
  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::unsubscribe,
				 pattern ); //@@ argument list
  
  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::removeAll( string pattern ) {
  
  if ( pattern == "" ) { pattern = "*"; }
  
  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::remove,
				 pattern ); //@@ argument list
  
  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::saveHistos( string name ) {

  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::save,
				 name ); //@@ argument list

  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::createSummaryHisto( sistrip::SummaryHisto histo, 
						     sistrip::SummaryType type, 
						     string top_level_dir,
						     sistrip::Granularity gran ) {
  
  if ( !histos_ ) { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to CommissioningHistograms!"; 
    return;
  }
  
  pair<sistrip::SummaryHisto,sistrip::SummaryType> summ0(histo,type);
  pair<string,sistrip::Granularity> summ1(top_level_dir,gran);
  seal::Callback action;
  action = seal::CreateCallback( histos_, 
				 &CommissioningHistograms::createSummaryHisto,
				 summ0, summ1 ); //@@ argument list
  
  if ( mui_ ) { 
    cout << endl // LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    cerr << endl // edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::uploadToConfigDb() {
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Derived implementation to come..."; 
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::subscribe( string pattern ) {
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]";
  if ( mui_ ) { mui_->subscribe(pattern); }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::update() {
  onUpdate();
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::unsubscribe( string pattern ) {
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]";
  if ( mui_ ) { mui_->unsubscribe(pattern); }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::remove( string pattern ) {
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]";
  if ( mui_ ) { mui_->getBEInterface()->rmdir(pattern); }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::save( string name ) {
  stringstream ss; 
  if ( name == "" ) { ss << "Client.root"; }
  else { ss << name; }
  cout << endl // LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Saving histogams to file '" << ss.str() << "'...";
  if ( mui_ ) { mui_->save( ss.str() ); }
}

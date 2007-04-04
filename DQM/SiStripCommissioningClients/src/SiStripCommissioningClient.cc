#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
// Commissioning tasks
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/FineDelayHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/DaqScopeModeHistograms.h"
#include "DQM/SiStripCommissioningClients/interface/HistogramDisplayHandler.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
// MessageLogger 
#include "FWCore/MessageService/interface/MessageLogger.h"
#include "EventFilter/Message2log4cplus/interface/MLlog4cplus.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
// 
#include <SealBase/Callback.h>
#include "cgicc/CgiDefs.h"
#include "cgicc/Cgicc.h"
#include "cgicc/CgiUtils.h"
#include "cgicc/HTTPResponseHeader.h"
#include "cgicc/HTMLClasses.h"

// This line is necessary
XDAQ_INSTANTIATOR_IMPL(SiStripCommissioningClient)

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningClient::SiStripCommissioningClient( xdaq::ApplicationStub* stub ) 
  : DQMBaseClient( stub, "SiStripCommissioningClient", "localhost", 9090 ),
    web_(0),
    histos_(0),
    runType_(sistrip::UNKNOWN_RUN_TYPE),
    first_(true),
    cfgFile_(""),
    handler_(0),
    presence_(),
    token_()
{

  xgi::bind( this, &SiStripCommissioningClient::handleWebRequest, "Request" );
  xgi::bind( this, &SiStripCommissioningClient::CBHistogramViewer, "HistogramViewer" );
  fCallBack = new BSem(BSem::EMPTY);
  fCallBack->give();
  hdis_ = NULL;

  // Retrieve configurables from xml configuration file
  xdata::InfoSpace* sp = getApplicationInfoSpace();
  sp->fireItemAvailable( "cfgFile", &cfgFile_ );

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
  // mui_->setVerbose(0);
  web_ = new SiStripCommissioningWebClient( this,
					    getContextURL(),
					    getApplicationURL(), 
					    &mui_ );
  hdis_ = new HistogramDisplayHandler(mui_,fCallBack);
  
  // Assert handler
  try {
    handler_ = new edm::AssertHandler();   
  } catch (...) { handleException( __func__ ); }
  
  // Make ParameterSet based on .cfg file 
  std::string param_set;
  bool log4cplus = parameterSetToString( cfgFile_.value_, param_set );
  if ( param_set.empty() ) { 
    std::stringstream ss;
    ss << endl
       << "[SiStripCommissioningClient::" << __func__ << "]"
       << " Unable to retrieve ParameterSet from .cfg file!";
    LOG4CPLUS_WARN( this->getApplicationLogger(), ss.str() );
    return; 
  }
  
  boost::shared_ptr<edm::ParameterSet> params; 
  boost::shared_ptr< std::vector<edm::ParameterSet> > pServiceSets;
  makeParameterSets( param_set, params, pServiceSets );

  // Create service token and 
  try {
    token_ = edm::ServiceRegistry::createSet( *pServiceSets );
  } catch (...) { handleException( __func__ ); }

  // Make the service available
  edm::ServiceRegistry::Operate operate( token_ );
  
  // Register this xdaq appl with the ML service
  try {
    if ( !log4cplus ) { edm::Service<edm::service::MessageLogger>(); }
    else { edm::Service<ML::MLlog4cplus>()->setAppl(this); }
  } catch (...) { handleException( __func__ ); }

  // Check that PresenceFactory and MessageLogger service exist
  try {
    edm::PresenceFactory* pf = edm::PresenceFactory::get();
    if ( pf ) {
      presence_ = boost::shared_ptr<edm::Presence>( pf->makePresence("MessageServicePresence").release() );
      if ( !presence_.get() ) {
	LOG4CPLUS_WARN( this->getApplicationLogger(),
			"Unable to retrieve MessageLogger service from PresenceFactory!" );
      }
    } else {
      LOG4CPLUS_WARN( this->getApplicationLogger(),
		      "Unable to retrieve PresenceFactory!" );
    }      
  } 
  catch (...) { handleException( __func__ ); }
  
  // First printout to MessageLogger service
  edm::LogVerbatim(mlDqmClient_) 
    << "SiStripCommissioningClient::" << __func__ << "]"
    << " Started MessageLogger Service...";
  
}
  
// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Enabled" state. */
void SiStripCommissioningClient::newRun() {
  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Starting new run...";
  ( this->upd_ )->registerObserver( this ); 
  subscribeAll(); 
  first_ = true;
}

// -----------------------------------------------------------------------------
/** Called whenever the client enters the "Halted" state. */
void SiStripCommissioningClient::endRun() {
  LogTrace(mlDqmClient_) 
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Halting present run...";
  removeAll();
  unsubscribeAll(); 
  
  if ( histos_ ) { delete histos_; histos_ = 0; }
  if ( hdis_ ) { delete hdis_; hdis_ = 0; }
  //if ( handler_ ) { delete handler_; handler_ = 0; }
  
}

// -----------------------------------------------------------------------------
/** Called by the "Updater" following each update. */
void SiStripCommissioningClient::onUpdate() const {
  
  // Check that DQM monitor user interface object exists
  if ( !mui_ ) {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!";
    return;
  }
  
  // Print number of updates
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Number of updates: " << mui_->getNumUpdates();

  // Retrieve list of added contents
  std::vector<std::string> contents;
  mui_->getAddedContents( contents ); 
  
  // Extract run type from added contents
  if ( runType_ == sistrip::UNKNOWN_RUN_TYPE ) { 
    runType_ = CommissioningHistograms::runType( mui_->getBEInterface(),
						 contents ); 
  }
  
  // Create histograms for given commissioning task
  createHistograms( runType_ );
  
  // Extract histograms based on added contents
  if ( histos_ ) { histos_->extractHistograms( contents ); }
  
  // Create collation histograms based on added contents
  if ( histos_ ) { histos_->createCollations( contents ); }
  
}

// -----------------------------------------------------------------------------
/** Create histograms for given commissioning task. */
void SiStripCommissioningClient::createHistograms( const sistrip::RunType& task ) const {

  // Check if object already exists
  if ( histos_ ) { return; }
  
  // Create corresponding "commissioning histograms" object 
  if      ( task == sistrip::APV_TIMING )         { histos_ = new ApvTimingHistograms( mui_ ); }
  else if ( task == sistrip::FED_CABLING )        { histos_ = new FedCablingHistograms( mui_ ); }
  else if ( task == sistrip::FED_TIMING )         { histos_ = new FedTimingHistograms( mui_ ); }
  else if ( task == sistrip::OPTO_SCAN )          { histos_ = new OptoScanHistograms( mui_ ); }
  else if ( task == sistrip::VPSP_SCAN )          { histos_ = new VpspScanHistograms( mui_ ); }
  else if ( task == sistrip::PEDESTALS )          { histos_ = new PedestalsHistograms( mui_ ); }
  else if ( task == sistrip::FINE_DELAY )         { histos_ = new FineDelayHistograms( mui_ ); }
  else if ( task == sistrip::DAQ_SCOPE_MODE )     { histos_ = new DaqScopeModeHistograms( mui_ ); }
  else if ( task == sistrip::UNDEFINED_RUN_TYPE ) { histos_ = 0; }
  else if ( task == sistrip::UNKNOWN_RUN_TYPE )   {
    histos_ = 0;
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Unknown commissioning task!";
  }
  
}

// -----------------------------------------------------------------------------
/** General access to client info. */
void SiStripCommissioningClient::general( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ) {
  if ( web_ ) { web_->Default( in, out ); }
  else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to WebPage!";
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::handleWebRequest( xgi::Input* in, xgi::Output* out ) throw ( xgi::exception::Exception ) {
  if ( web_ ) { web_->handleRequest(in, out); }
  else { 
    edm::LogWarning(mlDqmClient_)
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
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
    fCallBack->take();
    fCallBack->give();
  } else { 
    edm::LogWarning(mlDqmClient_)
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
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]" 
      << " NULL pointer to CommissioningHistograms!"; 
    return;
  }
  
  seal::Callback action; 
  action = seal::CreateCallback( histos_, 
				 &CommissioningHistograms::histoAnalysis,
				 debug ); // no arguments
  
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::subscribeAll( std::string pattern ) {

  if ( pattern == "" ) { pattern = "*/" + sistrip::root_ + "/*"; }

  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::subscribe,
				 pattern ); //@@ argument list
  
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
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
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::unsubscribeAll( std::string pattern ) {
  
  if ( pattern == "" ) { pattern = "*/" + sistrip::root_ + "/*"; }
  
  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::unsubscribe,
				 pattern ); //@@ argument list
  
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::removeAll( std::string pattern ) {
  
  if ( pattern == "" ) { pattern = "*"; }
  
  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::remove,
				 pattern ); //@@ argument list
  
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::saveHistos( std::string name ) {

  seal::Callback action;
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningClient::save,
				 name ); //@@ argument list

  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::createSummaryHisto( sistrip::Monitorable histo, 
						     sistrip::Presentation type, 
						     std::string top_level_dir,
						     sistrip::Granularity gran ) {
  
  if ( !histos_ ) { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to CommissioningHistograms!"; 
    return;
  }
  
  pair<sistrip::Monitorable,sistrip::Presentation> summ0(histo,type);
  pair<std::string,sistrip::Granularity> summ1(top_level_dir,gran);
  seal::Callback action;
  action = seal::CreateCallback( histos_, 
				 &CommissioningHistograms::createSummaryHisto,
				 summ0, summ1 ); //@@ argument list
  
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Scheduling this action...";
    mui_->addCallback(action); 
  } else { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
    return;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::uploadToConfigDb() {
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Derived implementation to come..."; 
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::subscribe( std::string pattern ) {
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Subscribing to all histograms within structure \""
      << pattern << "\"";
    mui_->subscribe(pattern); 
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::update() {
  LogTrace(mlDqmClient_)
    << "[SiStripCommissioningClient::" << __func__ << "]"
    << " Calling onUpdate() method...";
  onUpdate();
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::unsubscribe( std::string pattern ) {
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Unsubscribing to all histograms within structure \""
      << pattern << "\"";
    mui_->unsubscribe(pattern); 
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::remove( std::string pattern ) {
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Removing all histograms within structure \""
      << pattern << "\"";
    mui_->getBEInterface()->setVerbose(0);
    mui_->getBEInterface()->cd(); // cd to top dir
    mui_->getBEInterface()->removeContents(); 
    //if( mui_->getBEInterface()->dirExists(pattern) ) {
    //mui_->getBEInterface()->rmdir(pattern); 
    //}
    if( mui_->getBEInterface()->dirExists("Collector") ) {
      mui_->getBEInterface()->rmdir("Collector");
    }
    if( mui_->getBEInterface()->dirExists("EvF") ) {
      mui_->getBEInterface()->rmdir("EvF");
    }
    if( mui_->getBEInterface()->dirExists("SiStrip") ) {
      mui_->getBEInterface()->rmdir("SiStrip");
    }
    mui_->getBEInterface()->setVerbose(1);
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningClient::save( std::string name ) {
  std::stringstream ss; 
  if ( name == "" ) { ss << "Client.root"; }
  else { ss << name; }
  if ( mui_ ) { 
    LogTrace(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Saving all histograms to file \""
      << ss.str() << "\"";
    mui_->save( ss.str() ); 
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " NULL pointer to MonitorUserInterface!"; 
  }
}

// -----------------------------------------------------------------------------
// 
void SiStripCommissioningClient::handleException( const std::string& method,
						  const std::string& message ) {
  
  std::string name = "SiStripCommissioningClient::" + method;
  std::stringstream ss;

  try {
    //throw; // rethrow caught exception to be dealt with below
  } 

  catch ( seal::Error& e ) { 
    ss << " Caught seal::Error in method "
       << name << " with message: " << endl 
       << e.explainSelf();
    LOG4CPLUS_ERROR( this->getApplicationLogger(), ss.str() );
  }
  
  catch ( cms::Exception& e ) {
    ss << " Caught cms::Exception in method "
       << name << " with message: " << endl 
       << e.explainSelf();
    LOG4CPLUS_ERROR( this->getApplicationLogger(), ss.str() );
  }    
  
  catch ( std::exception& e ) {
    ss << " Caught std::exception in method "
       << name << " with message: " << endl 
       << e.what();
    LOG4CPLUS_ERROR( this->getApplicationLogger(), ss.str() );
  }
  
  catch (...) {
    ss << " Caught unknown exception in method "
       << name << " with no message";
    LOG4CPLUS_ERROR( this->getApplicationLogger(), ss.str() );
  }
  
}

// -----------------------------------------------------------------------------
// 
bool SiStripCommissioningClient::parameterSetToString( const std::string& cfg,
						       std::string& params ) {
  
  // Reset
  params.clear();

  // Environmetal variables
  std::string cmssw_base = "CMSSW_BASE";
  if ( getenv(cmssw_base.c_str()) ) { cmssw_base = getenv(cmssw_base.c_str()); }
  std::string cmssw_release = "CMSSW_RELEASE_BASE";
  if ( getenv(cmssw_release.c_str()) ) { cmssw_release = getenv(cmssw_release.c_str()); }
  
  // Find .cfg file for MessageLogger service
  std::string path; path.clear();
  if ( !cfg.empty() && !ifstream( cfg.c_str() ).fail() ) {
    path = cfg; 
  } else if ( !ifstream( (cmssw_base+"/src/DQM/SiStripCommon/data/Log4cplus.cfg").c_str() ).fail() ) { 
    path = cmssw_base+"/src/DQM/SiStripCommon/data/Log4cplus.cfg";
  } else if ( !ifstream( (cmssw_base+"/src/DQM/SiStripCommon/data/MessageLogger.cfg").c_str() ).fail() ) { 
    path = cmssw_base+"/src/DQM/SiStripCommon/data/MessageLogger.cfg";
  } else if ( !ifstream( (cmssw_release+"/src/DQM/SiStripCommon/data/MessageLogger.cfg").c_str() ).fail() ) { 
    path = cmssw_release+"/src/DQM/SiStripCommon/data/MessageLogger.cfg";
  } else {
    std::stringstream ss;
    ss << endl
       << "[SiStripCommissioningClient::" << __func__ << "]"
       << " No .cfg file has been found!";
    LOG4CPLUS_WARN( this->getApplicationLogger(), ss.str() );
  }
  
  // Read the ParameterSet from the .cfg file
  std::stringstream config; 
  if ( !path.empty() ) {
    ifstream in( path.c_str() );
    if( in.is_open() ) {
      while ( !in.eof() ) {
	std::string data;
	getline(in,data);
	config << data << "\n"; 
      }
      in.close();
      std::stringstream ss;
      ss << endl 
	 << "[SiStripCommissioningClient::" << __func__ << "]"
	 << " Found .cfg file at '" << path
	 << "' containing ParameterSet: " << endl
	 << config.str() << endl;
      LOG4CPLUS_INFO( this->getApplicationLogger(), ss.str() );
    }
  }  
  
  params = config.str();
  return ( config.str().find( "MLlog4cplus" ) != std::string::npos );
  
}


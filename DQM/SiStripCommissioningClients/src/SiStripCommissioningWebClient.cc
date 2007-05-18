#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningClient.h"
#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <SealBase/Callback.h>
#include <iostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningWebClient::SiStripCommissioningWebClient( SiStripCommissioningClient* client,
							      std::string context_url, 
							      std::string application_url, 
							      MonitorUserInterface** mui ) 
  : WebInterface( context_url, application_url, mui ),
    client_(client),
    mui_(*mui)
{
  defineWidgets();
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::defineWidgets() {
  
  // Define web page
  std::string url = this->getApplicationURL();
  page_p = new WebPage( url );
  
  // Commissioning-specific buttons 
  Button* subsc   = new Button( url, "20px", "20px", "SubscribeAll", "Subscribe all" );
  Button* unsub   = new Button( url, "50px", "20px", "UnsubscribeAll", "Unsubscribe all" );
  Button* update  = new Button( url, "80px", "20px", "UpdateHistos", "Update histograms" );
  Button* save    = new Button( url, "110px", "20px", "SaveHistos", "Save histos to file" );
  Button* anal    = new Button( url, "140px", "20px", "HistoAnalysis", "Analyze histograms" );
  Button* summary = new Button( url, "170px", "20px", "SummaryHisto", "Create summary histo" );
  Button* upload  = new Button( url, "200px", "20px", "UploadToDb", "Upload to database" );
  Button* remove  = new Button( url, "230px", "20px", "RemoveAll", "Remove all" );
  this->add( "SubscribeAll", subsc );
  this->add( "UnsubscribeAll", unsub );
  this->add( "UpdateHistos", update );
  this->add( "SaveHistos", save );
  this->add( "HistoAnalysis", anal );
  this->add( "SummaryHisto", summary );
  this->add( "UploadToDb", upload );
  this->add( "RemoveAll", remove );

  // Collector connection parameters, contents drop-down menu, viewer
  ContentViewer* con = new ContentViewer( url, "20px", "190px");
  ConfigBox* box = new ConfigBox( url, "20px", "340px");
  GifDisplay* dis = new GifDisplay( url, "260px", "20px", "500px", "700px", "GifDisplay" ); 
  add( "ConfigBox", box );
  add( "ContentViewer", con );
  add( "GifDisplay", dis );

}

// -----------------------------------------------------------------------------
/** Retrieve and handle std::strings that identify custom request(s). */
void SiStripCommissioningWebClient::handleCustomRequest( xgi::Input* in,
							 xgi::Output* out ) throw ( xgi::exception::Exception ) {
  
  // Retrieve requests
  CgiReader reader(in);
  std::multimap<std::string,std::string> requests;
  reader.read_form(requests);
  if ( requests.empty() ) { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Unable to handle empty request std::map!";
    return; 
  }
  
  std::string request = get_from_multimap( requests, "RequestID" );
  if ( request == "" ) { 
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Unable to handle empty request!";
    return; 
  }

  //@@ temporary
  std::string filename = "";
  sistrip::Monitorable mon = sistrip::APV_TIMING_DELAY;
  sistrip::Presentation pres = sistrip::SUMMARY_HISTO;
  std::string dir = "SiStrip/ControlView/FecCrate0/";
  sistrip::Granularity gran = sistrip::MODULE;
  
  // Handle requests
  if ( request == "SubscribeAll" ) { 
    if ( client_ ) { client_->subscribeAll( "*" ); }
  } else if ( request == "UnsubscribeAll" ) { 
    if ( client_ ) { client_->unsubscribeAll( "*" ); }
  } else if ( request == "UpdateHistos" ) { 
    if ( client_ ) { client_->onUpdate(); }
  } else if ( request == "SaveHistos" ) { 
    if ( client_ ) { client_->saveHistos( filename ); }
  } else if ( request == "HistoAnalysis" ) { 
    if ( client_ ) { client_->histoAnalysis( true ); }
  } else if ( request == "SummaryHisto" ) { 
    if ( client_ ) { client_->createSummaryHisto( mon, pres, dir, gran ); }
  } else if ( request == "UploadToDb" ) { 
    if ( client_ ) { client_->uploadToConfigDb(); }
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[SiStripCommissioningClient::" << __func__ << "]"
      << " Unknown request: " << request;
  }
  
}

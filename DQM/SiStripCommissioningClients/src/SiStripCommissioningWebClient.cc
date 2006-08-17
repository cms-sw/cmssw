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
#include <SealBase/Callback.h>
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningWebClient::SiStripCommissioningWebClient( SiStripCommissioningClient* client,
							      string context_url, 
							      string application_url, 
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
  string url = this->getApplicationURL();
  page_p = new WebPage( url );
  
  // Collector connection parameters, contents drop-down menu, viewer
  ContentViewer* con = new ContentViewer( url, "20px", "20px");
  ConfigBox* box = new ConfigBox( url, "20px", "340px");
  GifDisplay* dis = new GifDisplay( url, "170px", "20px", "500px", "700px", "GifDisplay" ); 
  add( "ConfigBox", box );
  add( "ContentViewer", con );
  add( "GifDisplay", dis );
  
  // Commissioning-specific buttons 
  Button* save    = new Button( url, "20px", "170px", "SaveHistos", "Save histos to file" );
  Button* summary = new Button( url, "60px", "170px", "SummaryHisto", "Create summary histo" );
  Button* tk_map  = new Button( url, "100px", "170px", "TrackerMap", "Create tracker map" );
  Button* upload  = new Button( url, "140px", "170px", "UploadToDb", "Upload to database" );
  this->add( "SaveHistos", save );
  this->add( "SummaryHisto", summary );
  this->add( "TrackerMap", tk_map );
  this->add( "UploadToDb", upload );

}

// -----------------------------------------------------------------------------
/** Retrieve and handle strings that identify custom request(s). */
void SiStripCommissioningWebClient::handleCustomRequest( xgi::Input* in,
							 xgi::Output* out ) throw ( xgi::exception::Exception ) {
  
  // Retrieve requests
  CgiReader reader(in);
  multimap<string,string> requests;
  reader.read_form(requests);
  if ( requests.empty() ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unable to handle empty request map!" 
	 << endl;
    return; 
  }
  
  // "Schedule" the request with the DQM using seal::Callback
  seal::Callback action; 
  action = seal::CreateCallback( this, 
				 &SiStripCommissioningWebClient::scheduleCustomRequest, 
				 requests ); // argument list
  if ( mui_ ) { mui_->addCallback(action); }
  else { cerr << "[" << __PRETTY_FUNCTION__ << "]" 
	      << " NULL pointer to MonitorUserInterface!" << endl; }

}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::scheduleCustomRequest( multimap<string,string> requests ) throw ( xgi::exception::Exception ) {
  
  string request = get_from_multimap( requests, "RequestID" );
  if ( request == "" ) { 
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unable to handle empty request!" 
	 << endl;
    return; 
  }

  // Retrieve pointer to histos object
  CommissioningHistograms* his = histos( *client_ );
  if ( !his ) {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL pointer to CommissioningHistograms!" 
	 << " Cannot handle request: " << request 
	 << endl;
    return;
  }
  
  // Handle requests
  if ( request == "SaveHistos" ) { 
    his->saveHistos("");
  } else if ( request == "SummaryHisto" ) { 
    his->createSummaryHisto( sistrip::APV_TIMING_DELAY,
			     sistrip::SUMMARY_SIMPLE_DISTR,
			     string("SiStrip/ControlView/FecCrate0/") );
  } else if ( request == "TrackerMap" ) {  
    his->createTrackerMap();
  } else if ( request == "UploadToDb" ) { 
    his->uploadToConfigDb();
  } else {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Unknown request: " << request 
	 << endl;
  }
  
}


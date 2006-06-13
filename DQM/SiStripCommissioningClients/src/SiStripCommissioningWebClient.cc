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
#include <iostream>
#include <map>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningWebClient::SiStripCommissioningWebClient( SiStripCommissioningClient* client,
							      string context_url, 
							      string application_url, 
							      MonitorUserInterface** mui ) 
  : WebInterface( context_url, application_url, mui ),
    client_(client)
{
  
  // Define web page
  string url = this->getApplicationURL();
  page_p = new WebPage( url );

  // Collector connection parameters
  ConfigBox* box = new ConfigBox( url, "20px", "20px");
  add( "ConfigBox", box );
  
  // Commissioning-specific buttons 
  Button* update  = new Button( url, "20px", "350px", "Update", "Refresh Client Histos" );
  Button* summary = new Button( url, "60px", "350px", "Summary", "Create Summary Histos" );
  Button* tk_map  = new Button( url, "100px", "350px", "TkMap", "Create Tracker Map" );
  Button* save    = new Button( url, "140px", "350px", "Save", "Save To File" );
  add( "Update", update );
  add( "Summary", summary );
  add( "TkMap", tk_map );
  add( "Save", save );

  // Contents drop-down menu
  ContentViewer* con = new ContentViewer( url, "200px", "20px");
  add( "ContentViewer", con );
  
  // Histogram viewer
  GifDisplay* dis = new GifDisplay( url, "650px", "20px", "400px", "600px", "GifDisplay" ); 
  add( "GifDisplay", dis );
  
}

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningWebClient::~SiStripCommissioningWebClient() {
}

// -----------------------------------------------------------------------------
/** Retrieve and handle strings that identify custom request(s). */
void SiStripCommissioningWebClient::handleCustomRequest( xgi::Input* in,
							 xgi::Output* out ) throw ( xgi::exception::Exception ) {
  CgiReader reader(in);
  multimap< string, string > requests;
  reader.read_form(requests);
  string request = get_from_multimap( requests, "RequestID" );
  if ( request == "Update" )  { updateHistos( in, out ); }
  if ( request == "Summary" ) { createSummary( in, out ); }
  if ( request == "TkMap" )   { createTkMap( in, out ); }
  if ( request == "Save" )    { saveToFile( in, out ); }
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::createSummary( xgi::Input* in, 
						   xgi::Output* out ) throw ( xgi::exception::Exception ) {
  cout << "[SiStripCommissioningWebClient::createSummary]" << endl;
  
  CommissioningHistograms* his = histo( *client_ );
  cout << "his " << his << endl;
  if ( his ) {
    cout << "here 1" << endl;
    his->createCollateMEs();
    his->createProfileHistos();
    his->createSummaryHistos();
    cout << "here 2" << endl;
  }
}


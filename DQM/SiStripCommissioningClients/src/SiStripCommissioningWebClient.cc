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
  
  // Define general stuff
  ConfigBox*     box = new ConfigBox( url, "50px", "50px");
  Navigator*     nav = new Navigator( url, "210px", "50px");
  ContentViewer* con = new ContentViewer( url, "340px", "50px");
  GifDisplay*    dis = new GifDisplay( url, "50px", "370px", "270px", "550px", "MyGifDisplay" ); 
  add( "ConfigBox", box );
  add( "Navigator", nav );
  add( "ContentViewer", con );
  add( "GifDisplay", dis );
  
  // Define commissioning-specific buttons 
  Button* summary = new Button( url, "400px", "150px", "CreateSummary", "Create Summary Histos" );
  Button* tk_map  = new Button( url, "440px", "150px", "CreateTkMap", "Create Tracker Map" );
  Button* save    = new Button( url, "480px", "150px", "SaveToFile", "Save To File" );
  add( "SummaryButton", summary );
  add( "TkMapButton", tk_map );
  add( "SaveButton", save );
  
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
  if ( request == "CreateSummary" ) { createSummary( in, out ); }
  if ( request == "CreateTkMap" )   { createTkMap( in, out ); }
  if ( request == "SaveToFile" )    { saveToFile( in, out ); }
  
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


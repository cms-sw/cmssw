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

  // Collector connection parameters, contents deop-down menu, viewer
  ConfigBox* box = new ConfigBox( url, "20px", "20px");
  ContentViewer* con = new ContentViewer( url, "20px", "350px");
  GifDisplay* dis = new GifDisplay( url, "220px", "20px", "600px", "800px", "GifDisplay" ); 
  add( "ConfigBox", box );
  add( "ContentViewer", con );
  add( "GifDisplay", dis );
  
  // Commissioning-specific buttons 
  Button* update  = new Button( url, "180px", "20px", "Update", "Refresh Histos" );
  Button* summary = new Button( url, "180px", "170px", "Summary", "Summary Histos" );
  Button* tk_map  = new Button( url, "180px", "320px", "TkMap", "Tracker Map" );
  Button* save    = new Button( url, "180px", "470px", "Save", "Save To File" );
  add( "Update", update );
  add( "Summary", summary );
  add( "TkMap", tk_map );
  add( "Save", save );
  
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
  if ( request != "" ) {
//     if ( request == "Update" )  { updateHistos( in, out ); }
//     if ( request == "Summary" ) { createSummary( in, out ); }
//     if ( request == "TkMap" )   { createTkMap( in, out ); }
//     if ( request == "Save" )    { saveToFile( in, out ); }
  }
}

// // -----------------------------------------------------------------------------
// /** */
// void SiStripCommissioningWebClient::updateHistos( xgi::Input* in, 
// 						  xgi::Output* out ) throw ( xgi::exception::Exception ) {
//   cout << "[SiStripCommissioningWebClient::updateHistos]"
//        << " Updating histograms..." << endl;
//   CommissioningHistograms* his = histos( *client_ );
//   if ( his ) {
//     his->createProfileHistos();
//   } else { cerr << "[SiStripCommissioningWebClient::updateHistos]"
// 		<< " Null pointer to CommissioningHistograms object!" << endl; }
//   cout << "[SiStripCommissioningWebClient::updateHistos]"
//        << " Updated histograms!" << endl;
// }

// // -----------------------------------------------------------------------------
// /** */
// void SiStripCommissioningWebClient::createSummary( xgi::Input* in, 
// 						   xgi::Output* out ) throw ( xgi::exception::Exception ) {
//   cout << "[SiStripCommissioningWebClient::createSummary]"
//        << " Creating summary histograms..." << endl;
//   CommissioningHistograms* his = histos( *client_ );
//   if ( his ) {
//     his->createProfileHistos();
//     his->createSummaryHistos();
//     cout << "[SiStripCommissioningWebClient::createSummary]"
// 	 << " Created summary histograms!" << endl;
//   }
// }

// // -----------------------------------------------------------------------------
// /** */
// void SiStripCommissioningWebClient::createTkMap( xgi::Input* in, 
// 						 xgi::Output* out ) throw ( xgi::exception::Exception ) {
//   cout << "[SiStripCommissioningWebClient::createTkMap]"
//        << " Creating Tracker map..." << endl;
//   CommissioningHistograms* his = histos( *client_ );
//   if ( his ) {
//     his->createProfileHistos();
//     cout << "[SiStripCommissioningWebClient::createTkMap]"
// 	 << " Created Tracker map!" << endl;
//   }
// }

// // -----------------------------------------------------------------------------
// /** */
// void SiStripCommissioningWebClient::saveToFile( xgi::Input* in, 
// 						xgi::Output* out ) throw ( xgi::exception::Exception ) {
//   cout << "[SiStripCommissioningWebClient::saveToFile]"
//        << " Saving histograms to file..." << endl;
//   CommissioningHistograms* his = histos( *client_ );
//   if ( his ) {
//     his->createProfileHistos();
//     his->createSummaryHistos();
//     //@@ create tracker map?
//     his->saveToFile();
//     cout << "[SiStripCommissioningWebClient::saveToFile]"
// 	 << " Saved histograms to file..." << endl;
//   }
// }

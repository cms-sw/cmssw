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
    client_(client),
    mui_(*mui)
{
  
  // Define web page
  string url = this->getApplicationURL();
  page_p = new WebPage( url );

  // Collector connection parameters, contents deop-down menu, viewer
  ContentViewer* con = new ContentViewer( url, "20px", "20px");
  ConfigBox* box = new ConfigBox( url, "20px", "340px");
  GifDisplay* dis = new GifDisplay( url, "170px", "20px", "500px", "700px", "GifDisplay" ); 
  add( "ConfigBox", box );
  add( "ContentViewer", con );
  add( "GifDisplay", dis );
  
  // Commissioning-specific buttons 
  Button* summary = new Button( url, "20px", "170px", "SummaryHistos", "Create summary histos" );
  Button* tk_map  = new Button( url, "60px", "170px", "TrackerMap", "Create tracker map" );
  Button* upload  = new Button( url, "100px", "170px", "UploadToDb", "Upload to database" );
  add( "SummaryHistos", summary );
  add( "TrackerMap", tk_map );
  add( "UploadToDb", upload );
  
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
    if ( request == "SummaryHistos" ) { createSummaryHistos( in, out ); }
    if ( request == "TrackerMap" )    { createTrackerMap( in, out ); }
    if ( request == "UploadToDb" )    { uploadToConfigDb( in, out ); }
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::createSummaryHistos( xgi::Input* in, 
							 xgi::Output* out ) throw ( xgi::exception::Exception ) {
  static const string method = "SiStripCommissioningWebClient::createSummaryHistos";
  cout << "["<<method<<"] Creating summary histograms..." << endl;
  
  // Retrieve pointer to commissioning histogram object
  CommissioningHistograms* his = histos( *client_ );
  if ( !his ) {
    cerr << "["<<method<<"] NULL pointer to CommissioningHistograms!" << endl;
    return;
  }
  
  // Summary histogram type and its directory
  vector<SummaryFactory::Histo> histos( 1, SummaryFactory::APV_TIMING_DELAY ); 
  string directory = "SiStrip/ControlView/FecCrate0/"; //@@ example
  
  // Extract view and directory level
  //   sistrip::View view = SiStripHistoNamingScheme::view( directory );
  //   SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath( directory ); 
  //   string summary_dir = SiStripHistoNamingScheme::controlPath( path ); 
  
  // Create summary histogram
  his->createSummaryHistos( histos, directory );
  
  cout << "["<<method<<"] Created summary histograms!" << endl;
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::createTrackerMap( xgi::Input* in, 
						      xgi::Output* out ) throw ( xgi::exception::Exception ) {
  cout << "[SiStripCommissioningWebClient::createTrackerMap]"
       << " Creating Tracker map..." << endl;
  CommissioningHistograms* his = histos( *client_ );
  if ( his ) {
    his->createTrackerMap();
    cout << "[SiStripCommissioningWebClient::createTrackerMap]"
	 << " Created Tracker map!" << endl;
  } else {
    cerr << "[SiStripCommissioningWebClient::createTrackerMap]"
	 << " NULL pointer to 'commissioning histograms' object!" << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::uploadToConfigDb( xgi::Input* in, 
						      xgi::Output* out ) throw ( xgi::exception::Exception ) {
  cout << "[SiStripCommissioningWebClient::uploadToConfigDb]"
       << " Uploading configurations to database..." << endl;
  CommissioningHistograms* his = histos( *client_ );
  if ( his ) {
    his->uploadToConfigDb();
    cout << "[SiStripCommissioningWebClient::uploadToConfigDb]"
	 << " Uploaded configurations to database!" << endl;
  } else {
    cerr << "[SiStripCommissioningWebClient::uploadToConfigDb]"
	 << " NULL pointer to 'commissioning histograms' object!" << endl;
  }
}

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
//#include <SealBase/Callback.h>
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
    mui_(*mui),
    action_(sistrip::NO_ACTION)
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
    if ( request == "SummaryHistos" ) { action_ = sistrip::CREATE_SUMMARY_HISTOS; }
    if ( request == "TrackerMap" )    { action_ = sistrip::CREATE_TRACKER_MAP; }
    if ( request == "UploadToDb" )    { action_ = sistrip::UPLOAD_TO_DATABASE; }
    else                              { action_ = sistrip::UNKNOWN_ACTION; }
  }

  //@@ how to move histo/type/view info to createSummaryHistos() method?

  // Schedules actions
  //   seal::Callback action( seal::CreateCallback( this, &SiStripCommissioningWebClient::performAction ) );
  //   mui_->addCallback( action );
  performAction(); //@@ temporarily here! should use Seal::Callback()!
  
}

// -----------------------------------------------------------------------------
/**  */
void SiStripCommissioningWebClient::performAction() {
  if      ( action_ == sistrip::CREATE_SUMMARY_HISTOS ) { createSummaryHistos(); }
  else if ( action_ == sistrip::CREATE_TRACKER_MAP )    { createTrackerMap(); }
  else if ( action_ == sistrip::UPLOAD_TO_DATABASE )    { uploadToConfigDb(); }
  else if ( action_ == sistrip::UNKNOWN_ACTION ) {
    cerr << "unknown action!" << endl;
  }
  action_ = sistrip::NO_ACTION;
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::createSummaryHistos() {
  static const string method = "SiStripCommissioningWebClient::createSummaryHistos";
  cout << "["<<method<<"] Creating summary histograms..." << endl;
  
  // Retrieve pointer to commissioning histogram object
  CommissioningHistograms* his = histos( *client_ );
  if ( !his ) {
    cerr << "["<<method<<"] NULL pointer to CommissioningHistograms!" << endl;
    return;
  }
  
  //@@ Example summary histogram type and its directory
  vector<sistrip::SummaryHisto> histos( 1, sistrip::APV_TIMING_DELAY ); 
  sistrip::SummaryType type = sistrip::SUMMARY_SIMPLE_DISTR;
  string directory = "SiStrip/ControlView/FecCrate0/";
  
  // Create summary histograms
  his->createSummaryHistos( histos, type, directory );
  
  cout << "["<<method<<"] Created summary histograms!" << endl;
}

// -----------------------------------------------------------------------------
/** */
void SiStripCommissioningWebClient::createTrackerMap() {
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
void SiStripCommissioningWebClient::uploadToConfigDb() {
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

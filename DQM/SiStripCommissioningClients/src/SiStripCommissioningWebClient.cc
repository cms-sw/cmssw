#include "DQM/SiStripCommissioningClients/interface/SiStripCommissioningWebClient.h"
// dqm
#include "DQMServices/CoreROOT/interface/DaqMonitorROOTBackEnd.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/WebPage.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
// #include "DQMServices/WebComponents/interface/WebElement.h"
// #include "DQMServices/WebComponents/interface/Button.h"
// std
#include <iostream>
#include <map>

using namespace std;

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningWebClient::SiStripCommissioningWebClient( string context_url, 
							      string application_url, 
							      MonitorUserInterface** mui ) 
  : WebInterface( context_url, application_url, mui ),
    webpage_(0) {

  // Define widgets
  string url = this->getApplicationURL();
  ConfigBox*     box = new ConfigBox( url, "50px", "50px");
  Navigator*     nav = new Navigator( url, "210px", "50px");
  ContentViewer* con = new ContentViewer( url, "340px", "50px");
  GifDisplay*    dis = new GifDisplay( url, "50px", "370px", "270px", "550px", "MyGifDisplay" ); 

  // Define web page
  webpage_ = new WebPage( url );
  webpage_->add( "ConfigBox", box );
  webpage_->add( "Navigator", nav );
  webpage_->add( "ContentViewer", con );
  webpage_->add( "GifDisplay", dis );
 
}

// -----------------------------------------------------------------------------
/** */
SiStripCommissioningWebClient::~SiStripCommissioningWebClient() {
  //if ( webpage_ ) { delete webpage_; }
}

void SiStripCommissioningWebClient::Default( xgi::Input* in, 
					     xgi::Output* out ) throw ( xgi::exception::Exception ) {
  CgiWriter writer( out, this->getContextURL() );
  writer.output_preamble();
  writer.output_head();
  if ( webpage_ ) { webpage_->printHTML( out ); }
  else {
    cerr << "[SiStripCommissioningClient::general]"
	 << "Null pointer for webpage!" << endl;
  }
  writer.output_finish();
}

// -----------------------------------------------------------------------------
/** Retrieve and handle strings that identify custom request(s). */
void SiStripCommissioningWebClient::handleCustomRequest( xgi::Input* in,
							 xgi::Output* out ) throw ( xgi::exception::Exception ) {
  CgiReader reader(in);
  multimap< string, string > requests;
  reader.read_form(requests);
  string request = get_from_multimap( requests, "RequestID" );
//   if ( request == "SubscribeAll" ) { subscribeAll( in, out ); }

}

// // -----------------------------------------------------------------------------
// /** */
// void SiStripCommissioningWebClient::subscribeAll( xgi::Input* in, 
// 						  xgi::Output* out ) throw ( xgi::exception::Exception ) {
//   cout << "[SiStripCommissioningWebClient::subscribeAll]" << endl;
//   (*(this->mui_p))->subscribe("*");
// }

// // -----------------------------------------------------------------------------
// /** */
// int SiStripCommissioningWebClient::getUpdates() {
//   MonitorUserInterface* mui = *(this->mui_p); // just to make code readable!
//   if ( !mui ) { return -1; }
//   mui->subscribeNew("*");
//   return mui->getNumUpdates();
// }
  

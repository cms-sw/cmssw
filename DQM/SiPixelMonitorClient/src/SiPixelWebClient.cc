#include "DQM/SiPixelMonitorClient/interface/SiPixelWebClient.h"


SiPixelWebClient::SiPixelWebClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090,       // the port at which the collector listens
		  5,          // the delay between reconnect attempts
		  false       // do not act as server
		  )
{
  // Instantiate a web interface:
  webInterface_p = new SiPixelWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
  xgi::bind(this, &SiPixelWebClient::handleWebRequest, "Request");
}

/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiPixelWebClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // the web interface should know what to do:
  webInterface_p->Default(in, out);
}


/*
  the method called on all HTTP requests of the form ".../Request?RequestID=..."
*/
void SiPixelWebClient::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
}

/*
  this obligatory method is called whenever the client enters the "Configured" state:
*/
void SiPixelWebClient::configure()
{

}

/*
  this obligatory method is called whenever the client enters the "Enabled" state:
*/
void SiPixelWebClient::newRun()
{
  upd_->registerObserver(this);
}

/*
  this obligatory method is called whenever the client enters the "Halted" state:
*/
void SiPixelWebClient::endRun()
{
}

/*
  this obligatory method is called by the Updater component, whenever there is an update 
*/
void SiPixelWebClient::onUpdate() const
{
  // put here the code that needs to be executed on every update:
  std::vector<std::string> uplist;
  mui_->getUpdatedContents(uplist);
}

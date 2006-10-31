#include "DQM/SiStripMonitorClient/interface/SiStripClient.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include <SealBase/Callback.h>

SiStripClient::SiStripClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
		  "localhost.cern.ch",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  )
{
  // Instantiate a web interface:
  webInterface_p = new SiStripWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
  xgi::bind(this, &SiStripClient::handleWebRequest, "Request");
}

/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiStripClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
   string link = getContextURL() + "/temporary/Online.html";
   webInterface_p->createButton(out, link);

  // the web interface should know what to do:
  webInterface_p->Default(in, out);
}


/*
  the method called on all HTTP requests of the form ".../Request?RequestID=..."
*/
void SiStripClient::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
}

/*
  this obligatory method is called whenever the client enters the "Configured" state:
*/
void SiStripClient::configure()
{
  cout << "SiStripClient::configure: Reading Configuration " << endl;
  webInterface_p->readConfiguration(updateFrequencyForTrackerMap_, 
				    updateFrequencyForSummary_);
}

/*
  this obligatory method is called whenever the client enters the "Enabled" state:
*/
void SiStripClient::newRun()
{
  upd_->registerObserver(this);
}

/*
  this obligatory method is called whenever the client enters the "Halted" state:
*/
void SiStripClient::endRun()
{
}

/*
  this obligatory method is called by the Updater component, whenever there is an update 
*/
void SiStripClient::onUpdate() const
{
  if (!mui_) return;
  int nUpdate = mui_->getNumUpdates();
  if (nUpdate == 0) mui_->subscribe("Collector/*");

  // put here the code that needs to be executed on every update:
  std::vector<std::string> uplist;
  mui_->getUpdatedContents(uplist);

  // Collation of Monitor Element
  if (nUpdate == 10) {
    webInterface_p->setActionFlag(SiStripWebInterface::Collate);
    seal::Callback action(seal::CreateCallback(webInterface_p, 
			&SiStripWebInterface::performAction));
    mui_->addCallback(action); 
  }
  
  // Set Up Quality Tests
  if (nUpdate == 20) webInterface_p->setupQTests();


  // Creation of Summary 
  if (nUpdate == 50) {
    webInterface_p->setActionFlag(SiStripWebInterface::Summary);
    seal::Callback action(seal::CreateCallback(webInterface_p, 
				 &SiStripWebInterface::performAction));
    mui_->addCallback(action);	 
  }	
  // Creation of TrackerMap
  if (updateFrequencyForTrackerMap_ != -1 ) {
    if (nUpdate > 1 && nUpdate%updateFrequencyForTrackerMap_ == 1) {
      webInterface_p->setActionFlag(SiStripWebInterface::TemporaryTkMap);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
				 &SiStripWebInterface::performAction));
      mui_->addCallback(action); 
    }
  }
}

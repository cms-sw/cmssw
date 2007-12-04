#include "DQM/SiPixelMonitorClient/interface/SiPixelWebClient.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/ANSIColors.h"
#include "DQMServices/ClientConfig/interface/QTestHandle.h"
#include <SealBase/Callback.h>


SiPixelWebClient::SiPixelWebClient(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "SiPixelClient",     // the name by which the collector identifies the client
		  "localhost.cern.ch",// the name of the computer hosting the collector
		  9090       // the port at which the collector listens
		  )
{
//  cout<<"entering WebClient constructor"<<endl;
  // Instantiate a web interface:
  webInterface_p = new SiPixelWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
  xgi::bind(this, &SiPixelWebClient::handleWebRequest, "Request");
  
  actionExecutor_ = new SiPixelActionExecutor();
  
//  cout<<"leaving WebClient constructor"<<endl;
}

/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiPixelWebClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
//  cout<<"entering general"<<endl;
  // the web interface should know what to do:
  webInterface_p->Default(in, out);
//  cout<<"leaving general"<<endl;
}


/*
  the method called on all HTTP requests of the form ".../Request?RequestID=..."
*/
void SiPixelWebClient::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
//  cout<<"entering handleWebRequest"<<endl;
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
//  cout<<"leaving handleWebRequest"<<endl;
}

/*
  this obligatory method is called whenever the client enters the "Configured" state:
*/
void SiPixelWebClient::configure()
{
//  cout << "SiPixelClient::configure: Reading Configuration " << endl;
  webInterface_p->readConfiguration(updateFrequencyForTrackerMap_, 
				    updateFrequencyForBarrelSummary_,
				    updateFrequencyForEndcapSummary_,
				    updateFrequencyForGrandBarrelSummary_,
				    updateFrequencyForGrandEndcapSummary_,
				    messageLimitForQTests_);
//  cout<<"leaving configure"<<endl;
}

/*
  this obligatory method is called whenever the client enters the "Enabled" state:
*/
void SiPixelWebClient::newRun()
{
//  cout<<"entering newRun"<<endl;
  upd_->registerObserver(this);
//  cout<<"leaving newRun"<<endl;
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
//  cout<<"entering onUpdate"<<endl;
  if (!mui_) return;
  int nUpdate = mui_->getNumUpdates();
  
  // SubscribeAll once:
  if (nUpdate == 0) mui_->subscribe("Collector/*");

  // Set Up Quality Tests once:
  if (nUpdate == 1) webInterface_p->setupQTests();

  // Initial Collation of all Monitor Elements in all Sources:
  if (nUpdate == 2){ //actionExecutor_->createCollation(mui_);
      webInterface_p->setActionFlag(SiPixelWebInterface::Collate);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
			    &SiPixelWebInterface::performAction));
      mui_->addCallback(action); 
  }
  
  // Initial Creation of Summary ME's: 
  if (nUpdate == 3){ //actionExecutor_->createSummary(mui_);
      webInterface_p->setActionFlag(SiPixelWebInterface::Summary);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
			    &SiPixelWebInterface::performAction));
      mui_->addCallback(action);
  }	 

  // Initial Check of Quality Test Results:
  if (nUpdate == 4){ //actionExecutor_->checkQTestResults(mui_);
      webInterface_p->setActionFlag(SiPixelWebInterface::QTestResult);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
			    &SiPixelWebInterface::performAction));
      mui_->addCallback(action);
  }
  
  // put here the code that needs to be executed on every update:
  std::vector<std::string> uplist;
  mui_->getUpdatedContents(uplist);

  // Frequent Updating of Summary ME's: 
//  if (updateFrequencyForBarrelSummary_ != -1 ) {
    if (nUpdate > 0 && nUpdate%299 == 0) {
      webInterface_p->setActionFlag(SiPixelWebInterface::Summary);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
			    &SiPixelWebInterface::performAction));
      mui_->addCallback(action);	 
    }
//  }	
  // Frequent Updating of Quality Test Results:
  if (nUpdate > 0 && nUpdate%300 == 0){ 
      webInterface_p->setActionFlag(SiPixelWebInterface::QTestResult);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
			    &SiPixelWebInterface::performAction));
      mui_->addCallback(action);
  }
/*  // Creation of TrackerMap
  if (updateFrequencyForTrackerMap_ != -1 && nUpdate > 30) {
    if (nUpdate%updateFrequencyForTrackerMap_ == 1) {
      webInterface_p->setActionFlag(SiPixelWebInterface::CreateTkMap);
      seal::Callback action(seal::CreateCallback(webInterface_p, 
				 &SiPixelWebInterface::performAction));
      mui_->addCallback(action); 
    }
  }
*/  
  // Save ME's into a file: 
  if (nUpdate%550 == 0) {
    webInterface_p->setActionFlag(SiPixelWebInterface::SaveData);
    seal::Callback action(seal::CreateCallback(webInterface_p, 
			  &SiPixelWebInterface::performAction));
    mui_->addCallback(action); 
  }  

 
//  cout<<"leaving onUpdate"<<endl;

}

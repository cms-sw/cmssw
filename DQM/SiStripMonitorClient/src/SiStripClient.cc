#include "DQM/SiStripMonitorClient/interface/SiStripClient.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"

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
  actionExecutor_ = 0;
}

/*
  implement the method that outputs the page with the widgets (declared in DQMBaseClient):
*/
void SiStripClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
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
  if (actionExecutor_ == 0) actionExecutor_ = new SiStripActionExecutor();
  actionExecutor_->readConfiguration(updateFrequencyForTrackerMap_, updateFrequencyForSummary_);
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
  if (actionExecutor_) delete actionExecutor_;
  actionExecutor_ = 0;
}

/*
  this obligatory method is called by the Updater component, whenever there is an update 
*/
void SiStripClient::onUpdate() const
{
  // put here the code that needs to be executed on every update:
  std::vector<std::string> uplist;
  mui_->getUpdatedContents(uplist);
  checkCustomRequests();
}
/* 
  check and perform custom actions
*/
void SiStripClient::checkCustomRequests() const {

  int nUpdate = mui_->getNumUpdates();
  if  (nUpdate == 5) actionExecutor_->setupQTests(mui_);
 
  // Check the customised action requests from the WebInterface
  SiStripWebInterface::SiStripActionType action_flg = webInterface_p->getActionFlag();
  switch (action_flg) {
  case SiStripWebInterface::Collate :
    {
      actionExecutor_->createCollation(mui_);
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      break;
    }
  case SiStripWebInterface::PersistantTkMap :
    {
      system("rm -rf tkmap_files_old/*.jpg; rm -rf  tkmap_files_old/*.svg");
      system("rm -rf tkmap_files_old");
      system("mv tkmap_files tkmap_files_old");
      system("mkdir -p tkmap_files");
      actionExecutor_->createTkMap(mui_);
      system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      break;
    }
  case SiStripWebInterface::TemporaryTkMap :
    {
      system("mkdir -p tkmap_files");
      system("rm -rf tkmap_files/*.jpg; rm -rf tkmap_files/*.svg");
      actionExecutor_->createTkMap(mui_);
      system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      break;
    }
  case SiStripWebInterface::Summary :
    {
      actionExecutor_->createSummary(mui_);
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      break;
    }
  case SiStripWebInterface::QTestResult :
    {
      actionExecutor_->checkQTestResults(mui_);
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      break;
    }
  case SiStripWebInterface::SaveData :
    {
      cout << " Saving Monitoring Elements " << endl;
      //  mui_->save("SiStripWebInterface.root", "Collector/Collated",90);
      mui_->save("SiStripWebInterface.root");
      webInterface_p->setActionFlag(SiStripWebInterface::NoAction);
      break;
    }
  case SiStripWebInterface::NoAction :
    {
      if (nUpdate > 5 && (nUpdate%updateFrequencyForTrackerMap_ == 0)) {
	system("mkdir -p tkmap_files");
	system("rm -rf tkmap_files/*.jpg; rm -rf tkmap_files/*.svg");
	actionExecutor_->createTkMap(mui_);
	system(" mv *.jpg tkmap_files/. ; mv *.svg tkmap_files/.");
      }
      break;
    }
  }
}
//
// -- Set Up Quality Test
//
void SiStripClient::setupQTest() const {
  cout << " Setting up Quality Tests " << endl;
  actionExecutor_->setupQTests(mui_);
  return;
}

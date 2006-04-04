#include "DQMServices/Examples/interface/ClientWithWebInterface.h"


ClientWithWebInterface::ClientWithWebInterface(xdaq::ApplicationStub *stub) 
  : DQMBaseClient(
		  stub,       // the application stub - do not change
		  "test",     // the name by which the collector identifies the client
		  "localhost",// the name of the computer hosting the collector
		  9090        // the port at which the collector listens
		  )
{
  // Instantiate a web interface:
  webInterface_p = new ExampleWebInterface(getContextURL(),getApplicationURL(), & mui_);
  
  xgi::bind(this, &ClientWithWebInterface::handleWebRequest, "Request");
}

void ClientWithWebInterface::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // the web interface should know what to do:
  webInterface_p->Default(in, out);
}

void ClientWithWebInterface::handleWebRequest(xgi::Input * in, xgi::Output * out)
{
  // the web interface should know what to do:
  webInterface_p->handleRequest(in, out);
}

void ClientWithWebInterface::configure()
{
}

void ClientWithWebInterface::newRun()
{
  upd_->registerObserver(this);
}

void ClientWithWebInterface::endRun()
{
}

void ClientWithWebInterface::onUpdate() const
{
  // put here the code that needs to be executed on every update:
  std::vector<std::string> uplist;
  mui_->getUpdatedContents(uplist);
}

#include "DQMServices/XdaqCollector/interface/DQMBaseClient.h"
#include "DQMServices/XdaqCollector/interface/Updater.h"
#include "DQMServices/XdaqCollector/interface/Updater.h"
#include "DQMServices/Core/interface/MonitorUIRoot.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include <iostream>
using std::cout; using std::endl;

DQMBaseClient::DQMBaseClient(xdaq::ApplicationStub *s, 
			     std::string name, 
			     std::string server, 
			     int port,
			     int reconnect_delay_secs,
			     bool actAsServer) 
  : dqm::StateMachine(s)
{
  contextURL = getApplicationDescriptor()->getContextDescriptor()->getURL();
  applicationURL = contextURL + "/" + getApplicationDescriptor()->getURN();

  mui_    = 0;
  upd_    = 0;
  name_   = name;
  server_ = server;
  port_   = port;
  reconnect_delay_secs_ = reconnect_delay_secs;
  actAsServer_ = actAsServer;

  xgi::bind(this, &DQMBaseClient::Default, "Default");
  xgi::bind(this, &DQMBaseClient::general, "general");

  dqm::StateMachine::bind("fsm");

  fireConfiguration(name_, server_, port_);
}

void DQMBaseClient::fireConfiguration(std::string name, std::string server, int port)
{
  xdata::InfoSpace *sp = getApplicationInfoSpace();
  sp->fireItemAvailable("serverHost",&server_);
  sp->fireItemAvailable("serverPort",&port_);
  sp->fireItemAvailable("subscribeList",&subs_);
  sp->fireItemAvailable("actAsServer",&actAsServer_);
  sp->fireItemAvailable("reconnectDelaySecs",&reconnect_delay_secs_);
}

void DQMBaseClient::Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict) << std::endl;
  *out << cgicc::html().set("lang", "en").set("dir","ltr") << std::endl;
  std::string url = getApplicationDescriptor()->getContextDescriptor()->getURL() + "/" + getApplicationDescriptor()->getURN();
  *out << "<frameset rows=\"300,90%\">" << std::endl;
  *out << "  <frame src=\"" << url << "/fsm" << "\">" << std::endl;
  *out << "  <frame src=\"" << url << "/general" << "\">" << std::endl;
  *out << "</frameset>" << std::endl;
  *out << cgicc::html() << std::endl;
}

void DQMBaseClient::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  *out << "General access to client info " << std::endl;
}

void DQMBaseClient::configureAction(toolbox::Event::Reference e) 
  throw (toolbox::fsm::exception::Exception)
{
  cout << "configureAction called " << endl;
  if(!mui_)
    {
      mui_ = new MonitorUIRoot(server_, port_, name_, reconnect_delay_secs_, actAsServer_);
    }
#if 0 // FIXME (LAT): Removed as a feature no longer available.
  for(unsigned int i = 0; i < subs_.size(); i++)
      mui_->subscribe(*((std::string*)subs_.elementAt(i)));
#endif

  configure();
}

void DQMBaseClient::enableAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{

  if(mui_)
    upd_ = new dqm::Updater(mui_);
  cout << "enableAction called " << endl;
  newRun();
}

    
void DQMBaseClient::suspendAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
  cout << "suspendAction called " << endl;
}


void DQMBaseClient::resumeAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
  cout << "resumeAction called " << endl;
}


void DQMBaseClient::haltAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
  cout << "haltAction called " << endl;
  upd_->setStopped();
  delete upd_;
  endRun();
}

void DQMBaseClient::nullAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  //this action has no effect. A warning is issued to this end
  LOG4CPLUS_WARN(this->getApplicationLogger(),
		    "Null action invoked");
}

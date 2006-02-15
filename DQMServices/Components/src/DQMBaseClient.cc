#include "DQMServices/Components/interface/DQMBaseClient.h"
#include "DQMServices/Components/interface/Updater.h"
#include "xgi/include/xgi/Method.h"

#include <iostream>

using namespace std;

DQMBaseClient::DQMBaseClient(xdaq::ApplicationStub *s, std::string name) : dqm::StateMachine(s),
									   mui_(0), 
									   upd_(0), name_(name),
							 server_("localhost"),
							 port_(9090)
{
  xgi::bind(this, &DQMBaseClient::Default, "Default");
  xgi::bind(this, &DQMBaseClient::general, "general");
  dqm::StateMachine::bind("fsm");
  xdata::InfoSpace *sp = getApplicationInfoSpace();
  sp->fireItemAvailable("serverHost",&server_);
  sp->fireItemAvailable("serverPort",&port_);
  sp->fireItemAvailable("subscribeList",&subs_);
}

#include "xgi/include/xgi/Utils.h"

void DQMBaseClient::Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict) << std::endl;
  *out << cgicc::html().set("lang", "en").set("dir","ltr") << std::endl;
  std::string url = "/";
  url += getApplicationDescriptor()->getURN();
  *out << "<head>"                                                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"     << endl;
  *out << "</head>"                                                  << endl;
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

#include "DQMServices/Components/interface/Updater.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

void DQMBaseClient::configureAction(toolbox::Event::Reference e) 
  throw (toolbox::fsm::exception::Exception)
{
  cout << "configureAction called " << endl;
  if(!mui_)
    {
      mui_ = new MonitorUIRoot(server_, port_, name_);
    }
  for(unsigned int i = 0; i < subs_.size(); i++)
    mui_->subscribe(*((string*)subs_.elementAt(i)));

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

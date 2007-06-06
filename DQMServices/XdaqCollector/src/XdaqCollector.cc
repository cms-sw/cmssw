#include "DQMServices/XdaqCollector/interface/XdaqCollector.h"

//
// provides factory method for instantion of HellWorld application
//
CollectorRoot *XdaqCollector::DummyConsumerServer::instance_=0;

XdaqCollector::XdaqCollector(xdaq::ApplicationStub * s) : dqm::StateMachine(s)
{	
  port_ = 9090;
  enableClients_ = false;
  xdata::InfoSpace *sp = getApplicationInfoSpace();
  sp->fireItemAvailable("listenPort",&port_);
  sp->fireItemAvailable("enableClients",&enableClients_);
  sp->fireItemAvailable("stateName",stateName());
  xgi::bind(this, &XdaqCollector::Default, "Default");
  xgi::bind(this, &XdaqCollector::general, "general");
  dqm::StateMachine::bind("fsm");
  xgi::bind(this, &XdaqCollector::css    , "styles.css");
}

#include "EventFilter/Utilities/interface/Css.h"
void XdaqCollector::css(xgi::Input  *in,
	 xgi::Output *out) throw (xgi::exception::Exception)

{
  evf::Css css;
  css.css(in,out);
}

void XdaqCollector::Default(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict)	<< std::endl;
  *out << cgicc::html().set("lang", "en").set("dir","ltr")	<< std::endl;
  std::string url = "/";
  url += getApplicationDescriptor()->getURN();
  *out << "<head>"                                              << std::endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"					<< std::endl;
  *out << "</head>"                                             << std::endl;
  *out << "<frameset rows=\"300,90%\">" << std::endl;
  *out << "  <frame src=\"" << url << "/fsm" << "\">"		<< std::endl;
  *out << "  <frame src=\"" << url << "/general" << "\">"	<< std::endl;
  *out << "</frameset>"						<< std::endl;
  *out << cgicc::html()						<< std::endl;
}

void XdaqCollector::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  std::string url = "/";
  url += getApplicationDescriptor()->getURN();
  *out << "<html>"						<< std::endl;
  *out << "<head>"						<< std::endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"" <<  url
       << "/styles.css\"/>"					<< std::endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"					<< std::endl;
  *out << "<META HTTP-EQUIV=refresh CONTENT=\"30; URL=";
  *out << url << "/general\">"					<< std::endl;
  *out << "</head>"                                             << std::endl;
  *out << "<body>"                                              << std::endl;
  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << std::endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"               << std::endl;
    *out << "  <tr>"                                            << std::endl;
    *out << "    <th colspan=2>"                                << std::endl;
    *out << "      " << "Configuration"                         << std::endl;
    *out << "    </th>"                                         << std::endl;
    *out << "  </tr>"                                           << std::endl;

    *out << "<tr>"						<< std::endl;
    *out << "<th >"						<< std::endl;
    *out << "Parameter"						<< std::endl;
    *out << "</th>"						<< std::endl;
    *out << "<th>"						<< std::endl;
    *out << "Value"						<< std::endl;
    *out << "</th>"						<< std::endl;
    *out << "</tr>"						<< std::endl;
    *out << "<tr>"						<< std::endl;
    *out << "<td >"						<< std::endl;
    *out << "Listen Port"					<< std::endl;
    *out << "</td>"						<< std::endl;
    *out << "<td>"						<< std::endl;
    *out << port_						<< std::endl;
    *out << "</td>"						<< std::endl;
    *out << "</tr>"						<< std::endl;  
    *out << "<tr>"						<< std::endl;
    *out << "<td >"						<< std::endl;
    *out << "Clients Enabled"					<< std::endl;
    *out << "</td>"						<< std::endl;
    *out << "<td>"						<< std::endl;
    *out << enableClients_					<< std::endl;
    *out << "</td>"						<< std::endl;
    *out << "  </tr>"						<< std::endl;  
    *out << "</table>"						<< std::endl;

    *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">" << std::endl;
    *out << "  <tr>"                                            << std::endl;
    *out << "    <th colspan=3>"                                << std::endl;
    *out << "      " << "Status"                                << std::endl;
    *out << "    </th>"                                         << std::endl;
    *out << "  </tr>"                                           << std::endl;

    *out << "<tr >"						<< std::endl;
    *out << "<th >"						<< std::endl;
    *out << "Type"						<< std::endl;
    *out << "</th>"						<< std::endl;
    *out << "<th >"						<< std::endl;
    *out << "Name"						<< std::endl;
    *out << "</th>"						<< std::endl;
    *out << "<th >"						<< std::endl;
    *out << "Received/Sent"					<< std::endl;
    *out << "</th>"						<< std::endl;
    *out << "</tr>"						<< std::endl;
    CollectorRoot *cl = DummyConsumerServer::instance();
    if(cl)
      {
	std::vector<std::string> sources = cl->getSourceNames();
	std::vector<std::string> clients = cl->getClientNames();
	for(unsigned int idesc = 0; idesc < sources.size(); idesc++)
	  {
	    *out << "<tr>"					<< std::endl;
	    *out << "<td >"					<< std::endl;
	    *out << "Source"					<< std::endl;
	    *out << "</td>"					<< std::endl;
	    *out << "<td >"					<< std::endl;
	    *out << sources[idesc]				<< std::endl;
	    *out << "</td>"					<< std::endl;
	    *out << "<td >"					<< std::endl;
	    *out << cl->getNumReceived(sources[idesc])		<< std::endl;
	    *out << "</td>"					<< std::endl;
	    *out << "</tr>"					<< std::endl;
	  }
	for(unsigned int idesc = 0; idesc < clients.size(); idesc++)
	  {
	    *out << "<tr>"					<< std::endl;
	    *out << "<td >"					<< std::endl;
	    *out << "Client"					<< std::endl;
	    *out << "</td>"					<< std::endl;
	    *out << "<td >"					<< std::endl;
	    *out << clients[idesc]				<< std::endl;
	    *out << "</td>"					<< std::endl;
	    *out << "<td >"					<< std::endl;
	    *out << cl->getNumSent(clients[idesc])		<< std::endl;
	    *out << "</td>"					<< std::endl;
	    *out << "</tr>"					<< std::endl;
	  }
      }
    else
      *out << "Unconfigured"					<< std::endl;
  *out << "</table>"						<< std::endl;
  *out << "</body>"                                             << std::endl;
  *out << "</html>"                                             << std::endl;
}

void XdaqCollector::configureAction(toolbox::Event::Reference e) 
  throw (toolbox::fsm::exception::Exception)
{
  DummyConsumerServer *dcs = dynamic_cast<DummyConsumerServer *> 
    (DummyConsumerServer::instance(port_) );
  if(dcs){} // trying to fix warning for unused variable
  //dcs->disableClients();
}
  
void XdaqCollector::enableAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
  DummyConsumerServer::start();
}   
void XdaqCollector::suspendAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
}

void XdaqCollector::resumeAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
}

void XdaqCollector::haltAction(toolbox::Event::Reference e) 
    throw (toolbox::fsm::exception::Exception)
{
  DummyConsumerServer::stopAndKill();
}

void XdaqCollector::nullAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  //this action has no effect. A warning is issued to this end
  LOG4CPLUS_WARN(this->getApplicationLogger(),
		    "Null action invoked");

}


void XdaqCollector::actionPerformed (xdata::Event& e)
{
  if (e.type() == "ItemChangedEvent" )
    {
      std::string item = dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
      DummyConsumerServer *dcs = 0;
      dcs = (DummyConsumerServer *)DummyConsumerServer::instance(0);
      /*
      if ( item == "enableClients")
	{
	  if(enableClients_)
	    {
	      if(dcs != 0)
	      // this modifies DummyConsumerServer::enableClients_
		dcs->enableClients(); 
	    }
	  else
	    {
	      if(dcs != 0)
	      // this modifies DummyConsumerServer::enableClients_
		dcs->disableClients();
	    }
	}
      */
    }
}


XDAQ_INSTANTIATOR_IMPL(XdaqCollector)


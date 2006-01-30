#include "DQMServices/XdaqCollector/interface/XdaqCollector.h"

//
// provides factory method for instantion of HellWorld application
//
CollectorRoot *XdaqCollector::DummyConsumerServer::instance_=0;

XdaqCollector::XdaqCollector(xdaq::ApplicationStub * s) : dqm::StateMachine(s)
{	
  port_ = 9090;
  xdata::InfoSpace *sp = getApplicationInfoSpace();
  sp->fireItemAvailable("listenPort",&port_);
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

void XdaqCollector::general(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  std::string url = "/";
  url += getApplicationDescriptor()->getURN();
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"" <<  url
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"     << endl;
  *out << "<META HTTP-EQUIV=refresh CONTENT=\"30; URL=";
  *out << url << "/general\">" << std::endl;
  *out << "</head>"                                                  << endl;
  *out << "<body>"                                                   << endl;
  *out << "<table frame=\"void\" rules=\"groups\" class=\"states\">" << std::endl;
  *out << "<colgroup> <colgroup align=\"rigth\">"                    << std::endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=2>"                                       << endl;
    *out << "      " << "Configuration"                                << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    *out << "<tr>" << std::endl;
    *out << "<th >" << std::endl;
    *out << "Parameter" << std::endl;
    *out << "</th>" << std::endl;
    *out << "<th>" << std::endl;
    *out << "Value" << std::endl;
    *out << "</th>" << std::endl;
    *out << "</tr>" << std::endl;
    *out << "<tr>" << std::endl;
    *out << "<td >" << std::endl;
    *out << "Listen Port" << std::endl;
    *out << "</td>" << std::endl;
    *out << "<td>" << std::endl;
    *out << port_ << std::endl;
    *out << "</td>" << std::endl;
    *out << "  </tr>" << std::endl;                   
    *out << "</table>" << std::endl;

    *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">" << std::endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=3>"                                       << endl;
    *out << "      " << "Status"                                       << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

    *out << "<tr >" << std::endl;
    *out << "<th >" << std::endl;
    *out << "Type" << std::endl;
    *out << "</th>" << std::endl;
    *out << "<th >" << std::endl;
    *out << "Name" << std::endl;
    *out << "</th>" << std::endl;
    *out << "<th >" << std::endl;
    *out << "Received/Sent" << std::endl;
    *out << "</th>" << std::endl;
    *out << "</tr>" << std::endl;
    CollectorRoot *cl = DummyConsumerServer::instance();
    if(cl)
      {
	std::vector<std::string> sources = cl->getSourceNames();
	std::vector<std::string> clients = cl->getClientNames();
	for(unsigned int idesc = 0; idesc < sources.size(); idesc++)
	  {
	    *out << "<tr>" << std::endl;
	    *out << "<td >" << std::endl;
	    *out << "Source" << std::endl;
	    *out << "</td>" << std::endl;
	    *out << "<td >" << std::endl;
	    *out << sources[idesc] << std::endl;
	    *out << "</td>" << std::endl;
	    *out << "<td >" << std::endl;
	    *out << cl->getNumReceived(sources[idesc]) << std::endl;
	    *out << "</td>" << std::endl;
	    *out << "</tr>" << std::endl;
	  }
	for(unsigned int idesc = 0; idesc < clients.size(); idesc++)
	  {
	    *out << "<tr>" << std::endl;
	    *out << "<td >" << std::endl;
	    *out << "Client" << std::endl;
	    *out << "</td>" << std::endl;
	    *out << "<td >" << std::endl;
	    *out << clients[idesc] << std::endl;
	    *out << "</td>" << std::endl;
	    *out << "<td >" << std::endl;
	    *out << cl->getNumSent(clients[idesc]) << std::endl;
	    *out << "</td>" << std::endl;
	    *out << "</tr>" << std::endl;
	  }
      }
    else
      *out << "Unconfigured" << std::endl;
  *out << "</table>" << std::endl;
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;
}

void XdaqCollector::configureAction(toolbox::Event::Reference e) 
  throw (toolbox::fsm::exception::Exception)
{
  DummyConsumerServer::instance(port_);
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

XDAQ_INSTANTIATOR_IMPL(XdaqCollector);


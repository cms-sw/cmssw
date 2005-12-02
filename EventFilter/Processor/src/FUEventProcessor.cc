#include "EventFilter/Processor/interface/FUEventProcessor.h"
#include "EventFilter/Processor/interface/EventProcessor.h"
#include "toolbox/include/TaskGroup.h"
#include "xgi/include/xgi/Method.h"

using namespace evf;

FUEventProcessor::FUEventProcessor(xdaq::ApplicationStub *s) : xdaq::Application(s), proc_(0), group_(0), fsm_(0), ah_(0)
{
  ah_ = new edm::AssertHandler();
  fsm_ = new EPStateMachine(getApplicationLogger());
  fsm_->init<evf::FUEventProcessor>(this);
  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  // default configuration
  ispace->fireItemAvailable("parameterSet",&offConfig_);
  ispace->fireItemAvailable("stateName",&fsm_->stateName_);
  // Bind web interface
  xgi::bind(this, &FUEventProcessor::css           , "styles.css");
  xgi::bind(this, &FUEventProcessor::defaultWebPage, "Default"   );
}
FUEventProcessor::~FUEventProcessor()
{
  if(proc_) delete proc_;
  delete fsm_;
  delete ah_;
}
void FUEventProcessor::configureAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_ = new EventProcessor(getApplicationDescriptor()->getInstance());
  group_ = new TaskGroup();
  proc_->initTaskGroup(group_);
  proc_->init(offConfig_.value_);
}

void FUEventProcessor::enableAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_->beginRun();
  proc_->activate();
}

void FUEventProcessor::suspendAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_->suspend();
}

void FUEventProcessor::resumeAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_->resume();
}

void FUEventProcessor::haltAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_->stopEventLoop();
  proc_->kill();
  //  group_->join();
  proc_->endRun();

  delete proc_;
}

#include "xoap/include/xoap/SOAPEnvelope.h"
#include "xoap/include/xoap/SOAPBody.h"
#include "xoap/include/xoap/domutils.h"

xoap::MessageReference FUEventProcessor::fireEvent(xoap::MessageReference msg)
  throw (xoap::exception::Exception)
{
  xoap::SOAPPart     part      = msg->getSOAPPart();
  xoap::SOAPEnvelope env       = part.getEnvelope();
  xoap::SOAPBody     body      = env.getBody();
  DOMNode            *node     = body.getDOMNode();
  DOMNodeList        *bodyList = node->getChildNodes();
  DOMNode            *command  = 0;
  std::string        commandName;
  
  for (unsigned int i = 0; i < bodyList->getLength(); i++)
    {
      command = bodyList->item(i);
      
      if(command->getNodeType() == DOMNode::ELEMENT_NODE)
	{
	  commandName = xoap::XMLCh2String(command->getLocalName());
	  return fsm_->processFSMCommand(commandName);
	}
    }
  
  XCEPT_RAISE(xoap::exception::Exception, "Command not found");
}

void FUEventProcessor::defaultWebPage (xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  getApplicationDescriptor()->getURN()
       << "/styles.css\"/>"                   << endl;
  *out << "<title>" << getApplicationDescriptor()->getClassName() 
       << getApplicationDescriptor()->getInstance() 
       << " MAIN</title>"     << endl;
  *out << "</head>"                                                  << endl;
  *out << "<body>"                                                   << endl;
    *out << "<table border=\"0\" width=\"100%\">"                      << endl;
    *out << "<tr>"                                                     << endl;
    *out << "  <td align=\"left\">"                                    << endl;
    *out << "    <img"                                                 << endl;
    *out << "     align=\"middle\""                                    << endl;
    *out << "     src=\"/daq/evb/examples/fu/images/fu64x64.gif\""     << endl;
    *out << "     alt=\"main\""                                        << endl;
    *out << "     width=\"64\""                                        << endl;
    *out << "     height=\"64\""                                       << endl;
    *out << "     border=\"\"/>"                                       << endl;
    *out << "    <b>"                                                  << endl;
    *out << getApplicationDescriptor()->getClassName() 
	 << getApplicationDescriptor()->getInstance()                  << endl;
    *out << "      " << fsm_->stateName_.toString()                    << endl;
    *out << "    </b>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/urn:xdaq-application:lid=3\">"             << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/xdaq/hyperdaq/images/HyperDAQ.jpg\""    << endl;
    *out << "       alt=\"HyperDAQ\""                                  << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                      << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "  </td>"                                                  << endl;
    *out << "  <td width=\"32\">"                                      << endl;
    *out << "    <a href=\"/" << getApplicationDescriptor()->getURN() 
	 << "/debug\">"                   << endl;
    *out << "      <img"                                               << endl;
    *out << "       align=\"middle\""                                  << endl;
    *out << "       src=\"/daq/evb/bu/images/debug32x32.gif\""         << endl;
    *out << "       alt=\"debug\""                                     << endl;
    *out << "       width=\"32\""                                      << endl;
    *out << "       height=\"32\""                                     << endl;
    *out << "       border=\"\"/>"                                     << endl;
    *out << "    </a>"                                                 << endl;
    *out << "  </td>"                                                  << endl;
    *out << "</tr>"                                                    << endl;
    *out << "</table>"                                                 << endl;

  *out << "<hr/>"                                                    << endl;
  *out << "<table>"                                                  << endl;
  *out << "<tr valign=\"top\">"                                      << endl;
  *out << "  <td>"                                                   << endl;

  if(proc_)
    proc_->taskWebPage(in,out);
  else
    *out << "Unconfigured" << endl;
  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;

  *out << "<textarea rows=" << 10 << " cols=30 scroll=yes>"          << endl;
  *out << offConfig_.value_                                          << endl;
  *out << "</textarea><P>"                                           << endl;
  
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;

}

XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)

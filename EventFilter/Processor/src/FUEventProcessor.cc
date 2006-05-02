#include "FWCore/Framework/interface/EventProcessor.h"
#include "EventFilter/Processor/interface/FUEventProcessor.h"
#include "toolbox/include/TaskGroup.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/PresenceFactory.h"
#include "xgi/include/xgi/Method.h"
using namespace evf;

FUEventProcessor::FUEventProcessor(xdaq::ApplicationStub *s) : xdaq::Application(s), 
outPut_(true), inputPrescale_(1), outputPrescale_(1),  outprev_(true), 
proc_(0), group_(0), fsm_(0), ah_(0)
{
  std::cout << "FUEventProcessor constructor" << std::endl;
  ah_ = new edm::AssertHandler();
  fsm_ = new EPStateMachine(getApplicationLogger());
  fsm_->init<evf::FUEventProcessor>(this);
  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  // default configuration
  ispace->fireItemAvailable("parameterSet",&offConfig_);
  ispace->fireItemAvailable("stateName",&fsm_->stateName_);
  ispace->fireItemAvailable("outputEnabled",&outPut_);
  ispace->fireItemAvailable("globalInputPrescale",&inputPrescale_);
  ispace->fireItemAvailable("globalOutputPrescale",&outputPrescale_);

  // Add infospace listeners for exporting data values
  getApplicationInfoSpace()->addItemChangedListener ("outputEnabled", this);
  getApplicationInfoSpace()->addItemChangedListener ("globalInputPrescale", this);
  getApplicationInfoSpace()->addItemChangedListener ("globalOutputPrescale", this);

  // Bind web interface
  xgi::bind(this, &FUEventProcessor::css           , "styles.css");
  xgi::bind(this, &FUEventProcessor::defaultWebPage, "Default"   );
  xgi::bind(this, &FUEventProcessor::moduleWeb     , "moduleWeb"    );

  // Load the message service plug-in
  boost::shared_ptr<edm::Presence> theMessageServicePresence;
  try {
    m_messageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
        makePresence("MessageServicePresence").release());
  } catch(seal::Error& e) {
    XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		 e.explainSelf());
  }
  //test it 
  edm::LogInfo("FUEventProcessor") << "started MessageLogger Service ";

}
FUEventProcessor::~FUEventProcessor()
{
  if(proc_) delete proc_;
  delete fsm_;
  delete ah_;
}

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"

void FUEventProcessor::configureAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{

  ParameterSetRetriever pr(offConfig_.value_);
  std::string configString = pr.getAsString();
  std::cout << "Using config string \n" << configString << std::endl;
  try{
    proc_ = new edm::EventProcessor(configString);
  }
  catch(seal::Error& e)
    {
      XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		   e.explainSelf());
    }
  catch(cms::Exception &e)
    {
      XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		   e.explainSelf());
    }    

  catch(std::exception &e)
    {
      XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		   e.what());
    }
  catch(...)
    {
      XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		   "Unknown Exception");
    }
  if(!outPut_) //proc_->toggleOutput();
  //  proc_->prescaleInput(inputPrescale_);
  //  proc_->prescaleOutput(outputPrescale_);
  outprev_=outPut_;
  proc_->setRunNumber(1);
}

void FUEventProcessor::enableAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_->runAsync();
  int sc = proc_->statusAsync();

  if(sc != 0)
    {
      ostringstream errorString;
      errorString << "EventProcessor::runAsync returned status code" << sc;
      XCEPT_RAISE (toolbox::fsm::exception::Exception, 
		   errorString.str());
    }
}

void FUEventProcessor::suspendAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  //  proc_->suspend();
  LOG4CPLUS_WARN(this->getApplicationLogger(),
		    "EP::suspend has no effect, please use FU::suspend instead");
}

void FUEventProcessor::resumeAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  //  proc_->resume();
  LOG4CPLUS_WARN(this->getApplicationLogger(),
		    "EP::resume has no effect, please use FU::resume to resume a run previously suspended using FU::suspend");

}

void FUEventProcessor::nullAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  //this action has no effect. A warning is issued to this end
  LOG4CPLUS_WARN(this->getApplicationLogger(),
		    "Null action invoked");

}

void FUEventProcessor::haltAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  proc_->shutdownAsync();
  //  proc_->kill();
  //  group_->join();

  proc_->endJob();

  delete proc_;
}

void FUEventProcessor::actionPerformed (xdata::Event& e)
{
  if (e.type() == "ItemChangedEvent" && !(fsm_->stateName_.toString()=="Halted"))
    {
      std::string item = dynamic_cast<xdata::ItemChangedEvent&>(e).itemName();
      if ( item == "outputEnabled")
	{
	  if(outprev_ != outPut_)
	    {
	      LOG4CPLUS_WARN(this->getApplicationLogger(),
			     (outprev_ ? "Disabling " : "Enabling ") << "global output");
	      //	      proc_->toggleOutput();
	      outprev_ = outPut_;
	    }
	}
      if ( item == "globalInputPrescale")
	{
	  //	  proc_->prescaleInput(inputPrescale_);
	  LOG4CPLUS_WARN(this->getApplicationLogger(),
			 "Setting global input prescale factor to" << inputPrescale_);

	}
      if ( item == "globalOutputPrescale")
	{
	  //	  proc_->prescaleOutput(outputPrescale_);
	  LOG4CPLUS_WARN(this->getApplicationLogger(),
			 "Setting global output prescale factor to" << outputPrescale_);

	}
    }
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
  std::string urn = getApplicationDescriptor()->getURN();
  *out << "<!-- base href=\"/" <<  urn
       << "\"> -->" << endl;
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;
  *out << "<link type=\"text/css\" rel=\"stylesheet\"";
  *out << " href=\"/" <<  urn
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
    *out << "    <a href=\"/" << urn 
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
    taskWebPage(in,out,urn);
  else
    *out << "Unconfigured" << endl;
  *out << "  </td>"                                                  << endl;
  *out << "</table>"                                                 << endl;

  *out << "<textarea rows=" << 10 << " cols=80 scroll=yes>"          << endl;
  *out << offConfig_.value_                                          << endl;
  *out << "</textarea><P>"                                           << endl;
  
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;

}




#include "extern/cgicc/linuxx86/include/cgicc/CgiDefs.h"
#include "extern/cgicc/linuxx86/include/cgicc/Cgicc.h"
#include "extern/cgicc/linuxx86/include/cgicc/FormEntry.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"
#include "DataFormats/Common/interface/ModuleDescription.h"

void FUEventProcessor::taskWebPage(xgi::Input *in, xgi::Output *out, 
				 const std::string &urn)
{

  evf::filter *filt = 0;
  ModuleWebRegistry *mwr = 0;
  edm::ServiceRegistry::Operate operate(proc_->getToken());
  std::vector<edm::ModuleDescription const*> descs_ = proc_->getAllModuleDescriptions();				
  try{
    if(edm::Service<ModuleWebRegistry>().isAvailable())
      mwr = edm::Service<ModuleWebRegistry>().operator->();
  }
  catch(...)
    { cout <<"exception when trying to get the service registry " << endl;}

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
  *out << "Processed Events" << std::endl;
  *out << "</td>" << std::endl;
  *out << "<td>" << std::endl;
  *out << proc_->totalEvents() << std::endl;
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  *out << "<tr>" << std::endl;
  *out << "<td >" << std::endl;
  *out << "Endpaths State" << std::endl;
  *out << "</td>" << std::endl;
  //  *out << "<td" << (sched_->inhibit_endpaths_ ? " bgcolor=\"red\">" : ">") << std::endl;
  //  *out <<  (sched_->inhibit_endpaths_ ? "disabled" : "enabled") << std::endl;
  *out << "N/A this version";
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  *out << "<tr>" << std::endl;
  *out << "<td >" << std::endl;
  *out << "Global Input Prescale" << std::endl;
  *out << "</td>" << std::endl;
  //  *out << "<td" << (sched_->global_input_prescale_!=1 ? " bgcolor=\"red\">" : ">") << std::endl;
  //  *out <<  sched_->global_input_prescale_ << std::endl;
  *out << "N/A this version";
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  *out << "<tr>" << std::endl;
  *out << "<td >" << std::endl;
  *out << "Global Output Prescale" << std::endl;
  *out << "</td>" << std::endl;
  //  *out << "<td" << (sched_->global_output_prescale_!=1 ? " bgcolor=\"red\">" : ">") << std::endl;
  //  *out <<  sched_->global_output_prescale_ << std::endl;
  *out << "N/A this version";
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  
  
  
  *out << "</table>" << std::endl;
  
  *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">" << std::endl;
  *out << "  <tr>"                                                   << endl;
  *out << "    <th colspan=3>"                                       << endl;
  *out << "      " << "Application"                                  << endl;
  
  if(descs_.size()>0)
    *out << " (Process name=" << descs_[0]->processName_ << ")"       << endl;
  
  
  
  *out << "    </th>"                                                << endl;
  *out << "  </tr>"                                                  << endl;
  
  *out << "<tr >" << std::endl;
  *out << "<th >" << std::endl;
  *out << "Module" << std::endl;
  *out << "</th>" << std::endl;
  *out << "<th >" << std::endl;
  *out << "Label" << std::endl;
  *out << "</th>" << std::endl;
  *out << "<th >" << std::endl;
  *out << "Version" << std::endl;
  *out << "</th>" << std::endl;
  *out << "</tr>" << std::endl;
  
  for(unsigned int idesc = 0; idesc < descs_.size(); idesc++)
    {
      *out << "<tr>" << std::endl;
      *out << "<td >" << std::endl;
      if(mwr && mwr->checkWeb(descs_[idesc]->moduleName_))
	*out << "<a href=\"/" << urn << "/moduleWeb?module=" << descs_[idesc]->moduleName_ << "\">" 
	     << descs_[idesc]->moduleName_ << "</a>" << std::endl;
      else
	*out << descs_[idesc]->moduleName_ << std::endl;
      *out << "</td>" << std::endl;
      *out << "<td >" << std::endl;
      *out << descs_[idesc]->moduleLabel_ << std::endl;
      *out << "</td>" << std::endl;
      *out << "<td >" << std::endl;
      *out << descs_[idesc]->versionNumber_ << std::endl;
      *out << "</td>" << std::endl;
      *out << "</tr>" << std::endl;
    }
  *out << "</table>" << std::endl;
  *out << "<table border=1 bgcolor=\"#CFCFCF\">" << std::endl;
  *out << "<tr>" << std::endl;
  if(filt)
    {
      //HLT summary status goes here
    }
  else
    {      
      *out << "<td >" << std::endl;
      *out << "No Filter Module" << std::endl;
      *out << "</td>" << std::endl;
    }
  *out << "</tr>" << std::endl;
  *out << "</table>" << std::endl;
  


}
void FUEventProcessor::moduleWeb(xgi::Input  *in, xgi::Output *out)
  throw (xgi::exception::Exception)
{
  using namespace cgicc;
  Cgicc cgi(in);
  vector<FormEntry> el1;
  cgi.getElement("module",el1);
  if(el1.size()!=0)
    {
      string modnam = el1[0].getValue();
      //      if(proc_)
	//proc_->moduleWebPage(in, out, modnam);
    }
}
XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)

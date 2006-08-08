#include "FWCore/Framework/interface/EventProcessor.h"
#include "EventFilter/Processor/interface/FUEventProcessor.h"
#include "toolbox/include/TaskGroup.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/PresenceFactory.h"
#include "xgi/include/xgi/Method.h"

namespace evf{
  namespace internal{
    void addService(vector<edm::ParameterSet>& adjust,
			  string const& service)
    {
      edm::ParameterSet newpset;
      newpset.addParameter<string>("@service_type",service);
      adjust.push_back(newpset);
      // Record this new ParameterSet in the Registry!
      // try not to
      //      pset::Registry::instance()->insertParameterSet(newpset);
    }

  // Add a service to the services list if it is not already there
    void addServiceMaybe(vector<edm::ParameterSet>& adjust,
			 string const& service)
    {
      typedef std::vector<edm::ParameterSet>::const_iterator Iter;
      for(Iter it = adjust.begin(); it != adjust.end(); ++it)
	{
	  string name = it->getParameter<std::string>("@service_type");
	  if (name == service) return;
	}
      addService(adjust, service);
    }
  }
}



using namespace evf;

#include <stdlib.h>

FUEventProcessor::FUEventProcessor(xdaq::ApplicationStub *s) : xdaq::Application(s), 
outPut_(true), inputPrescale_(1), outputPrescale_(1),  outprev_(true), 
							       proc_(0), group_(0), fsm_(0), ah_(0), serviceToken_(), servicesDone_(false), rmt_p(0)
{
  string xmlClass = getApplicationDescriptor()->getClassName();
  unsigned long instance = getApplicationDescriptor()->getInstance();
  LOG4CPLUS_INFO(this->getApplicationLogger(),
		 xmlClass << instance << " constructor");
  std::cout << "FUEventProcessor constructor" << std::endl;
  LOG4CPLUS_INFO(this->getApplicationLogger(),
		 "plugin path:" << getenv("SEAL_PLUGINS"));

  fsm_ = new EPStateMachine(getApplicationLogger());
  fsm_->init<evf::FUEventProcessor>(this);

  add_ = "localhost";
  port_ = 9090;
  del_ = 5000;
  rdel_ = 5;
  ostringstream ns;
  ns << "FU" << instance;
  nam_ = ns.str();

  xdata::InfoSpace *ispace = getApplicationInfoSpace();
  // default configuration
  ispace->fireItemAvailable("parameterSet",&offConfig_);
  ispace->fireItemAvailable("pluginPath",&seal_plugins_);
  ispace->fireItemAvailable("stateName",&fsm_->stateName_);
  ispace->fireItemAvailable("runNumber",&runNumber_);
  ispace->fireItemAvailable("outputEnabled",&outPut_);
  ispace->fireItemAvailable("globalInputPrescale",&inputPrescale_);
  ispace->fireItemAvailable("globalOutputPrescale",&outputPrescale_);
  ispace->fireItemAvailable("collectorAddr",&add_);
  ispace->fireItemAvailable("collectorPort",&port_);
  ispace->fireItemAvailable("collSendUs",&del_);
  ispace->fireItemAvailable("collReconnSec",&rdel_);
  ispace->fireItemAvailable("monSourceName",&nam_);

  // Add infospace listeners for exporting data values
  getApplicationInfoSpace()->addItemChangedListener ("outputEnabled", this);
  getApplicationInfoSpace()->addItemChangedListener ("globalInputPrescale", this);
  getApplicationInfoSpace()->addItemChangedListener ("globalOutputPrescale", this);
  //set sourceId_
  string xmlClass_ = getApplicationDescriptor()->getClassName();
  unsigned int instance_ = getApplicationDescriptor()->getInstance();
  ostringstream sourcename;
  sourcename << xmlClass_ << "_" << instance_;
  sourceId_ = sourcename.str();


  // Bind web interface
  xgi::bind(this, &FUEventProcessor::css           , "styles.css");
  xgi::bind(this, &FUEventProcessor::defaultWebPage, "Default"   );
  xgi::bind(this, &FUEventProcessor::moduleWeb     , "moduleWeb"    );

  //  logger_ = this->getApplicationLogger();
}
FUEventProcessor::~FUEventProcessor()
{
  if(proc_) delete proc_;
  delete fsm_;
  delete ah_;
}

#include "EventFilter/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Message2log4cplus/interface/MLlog4cplus.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"

void FUEventProcessor::configureAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{

  int retval = setenv("SEAL_PLUGINS",seal_plugins_.value_.c_str(),0);
  if(retval != 0)
    LOG4CPLUS_ERROR(this->getApplicationLogger(),
		    "Failed to set SEAL_PLUGINS search path ");
  LOG4CPLUS_INFO(this->getApplicationLogger(),
		 "plugin path:" << getenv("SEAL_PLUGINS"));

  // Load the message service plug-in
  if(!servicesDone_)
    {
      
      try {
	ah_ = new edm::AssertHandler();   
	LOG4CPLUS_INFO(this->getApplicationLogger(),
		       "Trying to create message service presence ");
	edm::PresenceFactory *pf = edm::PresenceFactory::get();
	LOG4CPLUS_INFO(this->getApplicationLogger(),
		       "presence factory pointer is " << (int) pf);
	if(pf != 0)
	  m_messageServicePresence = boost::shared_ptr<edm::Presence>(pf->makePresence("MessageServicePresence").release());
	else
	  LOG4CPLUS_ERROR(this->getApplicationLogger(),
			  "Unable to create message service presence ");
	
      } 
      catch(seal::Error& e) 
	{
	  LOG4CPLUS_ERROR(this->getApplicationLogger(),
			  e.explainSelf());
	}
      catch(cms::Exception &e)
	{
	  LOG4CPLUS_ERROR(this->getApplicationLogger(),
			  e.explainSelf());
	}    
      
      catch(std::exception &e)
	{
	  LOG4CPLUS_ERROR(this->getApplicationLogger(),
			  e.what());
	}
      catch(...)
	{
	  LOG4CPLUS_ERROR(this->getApplicationLogger(),
			  "Unknown Exception");
	}
    }

  //test it 
  edm::LogInfo("FUEventProcessor") << "started MessageLogger Service ";



  ParameterSetRetriever pr(offConfig_.value_);
  std::string configString = pr.getAsString();

  //  boost::shared_ptr<ParameterSet>          params; // change this name!
  //  makeParameterSets(configString, params, pServiceSets);
  if(!servicesDone_)
    {
      vector<edm::ParameterSet> pServiceSets;
      internal::addServiceMaybe(pServiceSets, "DaqMonitorROOTBackEnd");
      internal::addServiceMaybe(pServiceSets, "MLlog4cplus");
      internal::addServiceMaybe(pServiceSets, "MonitorDaemon");
      serviceToken_ = edm::ServiceRegistry::createSet(pServiceSets);
      servicesDone_ = true;
    }

  edm::ServiceRegistry::Operate operate(serviceToken_);
  try{
    rmt_p = edm::Service<MonitorDaemon>()->rmt(add_, port_, del_, nam_, rdel_);
    edm::Service<ML::MLlog4cplus>()->setAppl(this);
  }
  catch(...)
      { 
	LOG4CPLUS_INFO(this->getApplicationLogger(),"exception when trying to get service MonitorDaemon");
      }

  edm::LogInfo("FUEventProcessor") << "Using config string \n" << configString;

  try{

    proc_ = new edm::EventProcessor(configString, serviceToken_, edm::serviceregistry::kTokenOverrides);
    if(!outPut_) //proc_->toggleOutput();
      //  proc_->prescaleInput(inputPrescale_);
      //  proc_->prescaleOutput(outputPrescale_);
      proc_->enableEndPaths(outPut_);
    
    outprev_=outPut_;

    proc_->setRunNumber(runNumber_.value_);

    ModuleWebRegistry *mwr = 0;
    try{
      if(edm::Service<ModuleWebRegistry>().isAvailable())
	mwr = edm::Service<ModuleWebRegistry>().operator->();
    }
    catch(...)
      { 
	LOG4CPLUS_INFO(this->getApplicationLogger(),"exception when trying to get service ModuleWebRegistry");
      }
    if(mwr)
      mwr->publish(getApplicationInfoSpace());

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
  LOG4CPLUS_INFO(this->getApplicationLogger(),
		 "Finished with FUEventProcessor configuration ");
  
}

void FUEventProcessor::enableAction(toolbox::Event::Reference e) throw (toolbox::fsm::exception::Exception)
{
  int sc = 0;
  try
    {
      proc_->runAsync();
      sc = proc_->statusAsync();
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
  int trycount = 0;
  try
    {
      //      rmt_p->pause();
      sdn_.notify();
      edm::event_processor::State procstate = proc_->getState();
      while(proc_->getState() != edm::event_processor::sStopping && trycount < 10)
	{
	  trycount++;
	  ::sleep(1);
	}
      trycount = 0;
      if(proc_->getState() == edm::event_processor::sStopping)
	{
	  int retval = proc_->shutdownAsync();
	  while(proc_->getState() != edm::event_processor::sDone && trycount < 10)
	    {
	      trycount++;
	      ::sleep(1);
	    }
	}

      if(proc_->getState() != edm::event_processor::sDone)
	{
	  LOG4CPLUS_WARN(this->getApplicationLogger(),
			 "Halting with triggers still to be processed. EventProcessor state"
			 << proc_->stateName(proc_->getState()) );  
	  int retval = proc_->shutdownAsync();
	  //  proc_->kill();
	  //  group_->join();
	  if(retval != 0)
	    {
	      LOG4CPLUS_WARN(this->getApplicationLogger(),
			     "Failed to shut down EventProcessor. Return code " << retval);}	  
	  else
	    {
	      LOG4CPLUS_INFO(this->getApplicationLogger(),
			     "EventProcessor successfully shut down " << retval);
	    }
	}
      else
	LOG4CPLUS_INFO(this->getApplicationLogger(),
		       "EventProcessor halted. State" 
		       << proc_->stateName(proc_->getState()));  
      
      proc_->endJob();
      
      delete proc_;
      //      rmt_p->release();
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
  sdn_.cleanup();
    proc_ = 0;
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
	      proc_->enableEndPaths(outPut_);
	      outprev_ = outPut_;
	    }
	}
      if ( item == "globalInputPrescale")
	{
	  //	  proc_->prescaleInput(inputPrescale_);
	  //	  LOG4CPLUS_WARN(this->getApplicationLogger(),
	  //			 "Setting global input prescale factor to" << inputPrescale_);
	  //
	  LOG4CPLUS_WARN(this->getApplicationLogger(),
			 "Setting global input prescale has no effect in this version of the code");
	  
	  
	}
      if ( item == "globalOutputPrescale")
	{
	  //	  proc_->prescaleOutput(outputPrescale_);
	  //LOG4CPLUS_WARN(this->getApplicationLogger(),
	  //			 "Setting global output prescale factor to" << outputPrescale_);
	  LOG4CPLUS_WARN(this->getApplicationLogger(),
			 "Setting global output prescale has no effect in this version of the code");

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
  *out << "Processed Events/Accepted Events" << std::endl;
  *out << "</td>" << std::endl;
  *out << "<td>" << std::endl;
  *out << proc_->totalEvents() << "/" << proc_->totalEventsPassed() << std::endl;
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  *out << "<tr>" << std::endl;
  *out << "<td >" << std::endl;
  *out << "Endpaths State" << std::endl;
  *out << "</td>" << std::endl;
  *out << "<td";
  *out << (proc_->endPathsEnabled() ?  "> enabled" : 
	   " bgcolor=\"red\"> disabled" ) << std::endl;
  //*out << "> N/A this version" << std::endl;
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  *out << "<tr>" << std::endl;
  *out << "<td >" << std::endl;
  *out << "Global Input Prescale" << std::endl;
  *out << "</td>" << std::endl;
  *out << "<td";
  //*out << (sched_->global_input_prescale_!=1 ? " bgcolor=\"red\">" : ">") << std::endl;
  //  *out <<  sched_->global_input_prescale_ << std::endl;
  *out << "> N/A this version" << std::endl;
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  *out << "<tr>" << std::endl;
  *out << "<td >" << std::endl;
  *out << "Global Output Prescale" << std::endl;
  *out << "</td>" << std::endl;
  *out << "<td";
  //*out  << (sched_->global_output_prescale_!=1 ? " bgcolor=\"red\">" : ">") << std::endl;
  //  *out <<  sched_->global_output_prescale_ << std::endl;
  *out << ">N/A this version" << std::endl;
  *out << "</td>" << std::endl;
  *out << "  </tr>"                                            << endl;
  
  
  
  *out << "</table>" << std::endl;
  
  *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">" << std::endl;
  *out << "  <tr>"                                                   << endl;
  *out << "    <th colspan=3>"                                       << endl;
  *out << "      " << "Application"                                  << endl;
  
  if(descs_.size()>0)
    *out << " (Process name=" << descs_[0]->processName() << ")"       << endl;
  
  
  
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
      if(mwr && mwr->checkWeb(descs_[idesc]->moduleName()))
	*out << "<a href=\"/" << urn << "/moduleWeb?module=" << descs_[idesc]->moduleName() << "\">" 
	     << descs_[idesc]->moduleName() << "</a>" << std::endl;
      else
	*out << descs_[idesc]->moduleName() << std::endl;
      *out << "</td>" << std::endl;
      *out << "<td >" << std::endl;
      *out << descs_[idesc]->moduleLabel() << std::endl;
      *out << "</td>" << std::endl;
      *out << "<td >" << std::endl;
      *out << descs_[idesc]->releaseVersion() << std::endl;
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
  if(proc_)
    {
      if(el1.size()!=0)
	{
	  string mod = el1[0].getValue();
	  edm::ServiceRegistry::Operate operate(proc_->getToken());
	  ModuleWebRegistry *mwr = 0;
	  try{
	    if(edm::Service<ModuleWebRegistry>().isAvailable())
	      mwr = edm::Service<ModuleWebRegistry>().operator->();
	  }
	  catch(...)
	    { 
	      cout <<"exception when trying to get the service registry " << endl;
	    }
	  mwr->invoke(in,out,mod);
	}
    }
  else
    {
      *out << "EventProcessor just disappeared " << endl;
    }
}
XDAQ_INSTANTIATOR_IMPL(evf::FUEventProcessor)

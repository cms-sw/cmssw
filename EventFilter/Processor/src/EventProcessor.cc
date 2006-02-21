#include "EventFilter/Processor/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/src/InputSourceFactory.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include<iostream>
#include<string>

using namespace std;
namespace evf
{
  /* to be filled in with summary from paths */
  struct filter{
  };

  boost::shared_ptr<edm::InputSource> makeInput(edm::ParameterSet const& params_,
						const string& pname, 
						unsigned long pass,
						edm::ProductRegistry& preg)
  {
    // find single source
    try {
      const std::string& processName = params_.getParameter<string>("@process_name");

      edm::ParameterSet main_input = params_.getParameter<edm::ParameterSet>("@main_input");

      // Fill in "ModuleDescription", in case the input source produces any EDproducts,
      // which would be registered in the ProductRegistry.
      edm::ModuleDescription md;
      md.pid = main_input.id();
      md.moduleName_ = main_input.template getParameter<std::string>("@module_type");
      // There is no module label for the unnamed input source, so just use the module name.
      md.moduleLabel_ = md.moduleName_;
      md.processName_ = processName;
//#warning version and pass are hardcoded
      md.versionNumber_ = 1;
      md.pass = 1; 
      edm::InputSourceDescription isdesc(pname,pass,preg);
      boost::shared_ptr<edm::InputSource> input_
	(edm::InputSourceFactory::get()->makeInputSource(main_input, isdesc).release());
      input_->addToRegistry(md);
      return input_;
    } catch(const edm::Exception& iException) {
      if(edm::errors::Configuration == iException.categoryCode()) {
	throw edm::Exception(edm::errors::Configuration, "FailedInputSource")
          <<"Error in configuration of main input source.\n"
	  << iException;
      } else {
	throw;
      }
    }
    return boost::shared_ptr<edm::InputSource>();
  }
  

  
}
using namespace evf;

EventProcessor::EventProcessor(unsigned long tid) : 
  Task("FPTq"), /*input_(is),*/ tid_(tid), activityRegistry_(new edm::ActivityRegistry), serviceToken_(edm::ServiceToken()), 
  emittedBeginJob_(false), running_(false), paused_(false), exited_(false),
  eventcount(0)
{
  pthread_mutex_init(&mutex_,0);
  pthread_cond_init(&exit_,0);
}

EventProcessor::~EventProcessor()
{

}

#include "FWCore/ParameterSet/interface/MakeParameterSets.h"

void EventProcessor::init(std::string &config) 
{

  edm::ServiceToken iToken;
  edm::serviceregistry::ServiceLegacy iLegacy = edm::serviceregistry::kOverlapIsError;

  boost::shared_ptr< std::vector<edm::ParameterSet> > pServiceSets;

  //make the parameter set and the service sets from the config string 
  makeParameterSets(config, params_, pServiceSets);

  //create the services
  try{
    serviceToken_ = edm::ServiceRegistry::createSet(*pServiceSets,
						  iToken,iLegacy);
  }
  catch(edm::Exception &e)
    {
      cerr << "Exception when trying to create service set " 
	   << e.what() << endl;
      exit(-1);
    }

  serviceToken_.connectTo(*activityRegistry_);

  //add the ProductRegistry as a service ONLY for the construction phase
  boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::ConstProductRegistry> > 
    reg(new edm::serviceregistry::ServiceWrapper<edm::ConstProductRegistry>( 
								  std::auto_ptr<edm::ConstProductRegistry>(new edm::ConstProductRegistry(preg_))));
  edm::ServiceToken tempToken( edm::ServiceRegistry::createContaining(reg, serviceToken_, edm::serviceregistry::kOverlapIsError));
  
  //literally from FWCore/Framework/src/EventProcessor.cc
  // the next thing is ugly: pull out the trigger path pset and 
  // create a service and extra token for it
  string proc_name = params_->getParameter<string>("@process_name");
  
  typedef edm::service::TriggerNamesService TNS;
  typedef edm::serviceregistry::ServiceWrapper<TNS> w_TNS;
  
  edm::ParameterSet trigger_paths =
    (*params_).getUntrackedParameter<edm::ParameterSet>("@trigger_paths");
  boost::shared_ptr<w_TNS> tnsptr
    (new w_TNS( std::auto_ptr<TNS>(new TNS(trigger_paths,proc_name))));
  edm::ServiceToken tempToken2(edm::ServiceRegistry::createContaining(tnsptr, 
							      tempToken, 
							      edm::serviceregistry::kOverlapIsError));

  //make the services available
  edm::ServiceRegistry::Operate operate(tempToken2);

  act_table_ = edm::ActionTable(*params_);

  input_= makeInput(*params_, proc_name,
		    getPass(), preg_);


  //replaces ScheduleBuilder/ScheduleExecutor in the main function of steering 
  sched_ = std::auto_ptr<Schedule>(new Schedule(*params_,wreg_,
							  preg_,act_table_,
							  activityRegistry_));


  using namespace std;
  using namespace edm::eventsetup;
  vector<string> providers = (*params_).getParameter<vector<string> >("@all_esmodules");
  for(vector<string>::iterator itName = providers.begin();
      itName != providers.end();
      ++itName) {
    edm::ParameterSet providerPSet = (*params_).getParameter<edm::ParameterSet>(*itName);
    ModuleFactory::get()->addTo(esp_, 
				providerPSet, 
				proc_name, 
				getVersion(), 
				getPass());
  }

  vector<string> sources = (*params_).getParameter<vector<string> >("@all_essources");
  for(vector<string>::iterator itName = sources.begin();
      itName != sources.end();
      ++itName) {
    edm::ParameterSet providerPSet = (*params_).getParameter<edm::ParameterSet>(*itName);
    edm::eventsetup::SourceFactory::get()->addTo(esp_, 
						 providerPSet, 
						 (*params_).getParameter<string>("@process_name"), 
						 getVersion(), 
						 getPass());
  }

  //  fillEventSetupProvider(esp_, *params_, common_);  
}


#include "boost/bind.hpp"
#include "boost/mem_fn.hpp"
  //need a wrapper to let me 'copy' references to EventSetup
  namespace eventprocessor {
     struct ESRefWrapper {
        edm::EventSetup const & es_;
        ESRefWrapper(edm::EventSetup const &iES) : es_(iES) {}
        operator const edm::EventSetup&() { return es_; }
     };
  }
using eventprocessor::ESRefWrapper;
void EventProcessor::beginRun() 
{
  eventcount = 0;
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken_);
  running_=true;
  if(! emittedBeginJob_) {
    edm::EventSetup const& es = esp_.eventSetupForInstance(edm::IOVSyncValue::beginOfTime());
    Schedule::AllWorkers::iterator i(sched_->all_workers_.begin()),e(sched_->all_workers_.end());
    for(; i!=e; ++i) {
      descs_.push_back((*i)->description());
      
    }
    sched_->beginJob(es);
    emittedBeginJob_ = true;
    activityRegistry_->postBeginJobSignal_();
  }
}
#include <sys/time.h>
void EventProcessor::stopEventLoop(unsigned int delay)
{
  struct timeval now;
  struct timespec timeout;
  int retcode = 0;
  gettimeofday(&now,0);
  timeout.tv_sec = now.tv_sec + delay; //allow two seconds timeout before ending run
  timeout.tv_nsec = now.tv_usec * 1000;
  pthread_mutex_lock(&mutex_);
  running_ = false;
  retcode = pthread_cond_timedwait(&exit_,&mutex_,&timeout);
  pthread_mutex_unlock(&mutex_);
  if(retcode == ETIMEDOUT)
    {
      kill();
    }
}

void EventProcessor::toggleOutput()
{
  sched_->toggleEndPaths();
}

void EventProcessor::prescaleInput(unsigned int f)
{
  sched_->setGlobalInputPrescaleFactor(f);
}

void EventProcessor::prescaleOutput(unsigned int f)
{
  sched_->setGlobalOutputPrescaleFactor(f);
}


#include "FWCore/MessageLogger/interface/MessageLogger.h"

bool EventProcessor::endRun() 
{
  bool returnValue = true;

  if(running_)
    {    
      returnValue = false;
    }
  else
    {
      //make the services available
      edm::ServiceRegistry::Operate operate(serviceToken_);
          try
      {
	sched_->endJob();
	returnValue=true;
      }
    catch(cms::Exception& iException)
      {
	edm::LogError(iException.category())
	  << "Caught cms::Exception in endRun: "<< iException.what() << "\n";
      }
    catch(std::exception& iException)
      {
	edm::LogError("std::exception")
	  << "Caught std::exception in endRun: "<< iException.what() << "\n";
      }
    catch(...)
      {
	edm::LogError("ignored_exception")
	  << "Caught unknown exception in endRun. (ignoring it!!!!)\n";
      }
    }
  
  activityRegistry_->postEndJobSignal_();
  descs_.clear();
  return returnValue;
}
  


/**
 */                                             
#include "FWCore/Framework/interface/Event.h"

void EventProcessor::run()
{

  int oldState;
  pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, &oldState);

  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken_);
  if(exited_) exited_ = false;
  while(running_)
    {
      //      pthread_testcancel();
      if(paused_) 
	{
	  pause();
	}
      auto_ptr<edm::EventPrincipal> pep = input_->readEvent();
      ++eventcount;
      if(pep.get()==0) break;
      edm::IOVSyncValue ts(pep->id(), pep->time());
      edm::EventSetup const& es = esp_.eventSetupForInstance(ts);
      
      try
	{

	  //	  edm::EventRegistry::Operate oper(pep->id(),pep.get());
	  sched_->runOneEvent(*pep.get(),es);
	}
      catch(cms::Exception& e)
	{
	  edm::actions::ActionCodes code = act_table_.find(e.rootCause());
	  if(code==edm::actions::IgnoreCompletely)
	    {
	      // change to error logger!
	      cerr << "Ignoring exception from Event ID=" << pep->id()
		   << ", message:\n" << e.what()
		   << endl;
	      continue;
	    }
	  else if(code==edm::actions::SkipEvent)
	    {
	      cerr << "Skipping Event ID=" << pep->id()
		   << ", message:\n" << e.what()
		   << endl;
	      continue;
	    }
	  else
	    throw edm::Exception(edm::errors::EventProcessorFailure,
				 "EventProcessingStopped",e);
	}
      catch(seal::Error &e)
	{
	  cerr << e.explainSelf() << endl;
	  throw;
	}
    }
  pthread_mutex_lock(&mutex_);
  pthread_cond_signal(&exit_);
  pthread_mutex_unlock(&mutex_);
  exited_ = true;
}

#include "extern/cgicc/linuxx86/include/cgicc/CgiDefs.h"
#include "extern/cgicc/linuxx86/include/cgicc/Cgicc.h"
#include "extern/cgicc/linuxx86/include/cgicc/HTTPHTMLHeader.h"
#include "extern/cgicc/linuxx86/include/cgicc/HTMLClasses.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"


void EventProcessor::taskWebPage(xgi::Input *in, xgi::Output *out, 
				 const std::string &urn)
{

  evf::filter *filt = 0;
  edm::ServiceRegistry::Operate operate(serviceToken_);
  ModuleWebRegistry *mwr = 0;

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
	*out << eventcount << std::endl;
	*out << "</td>" << std::endl;
	*out << "  </tr>"                                            << endl;
	*out << "<tr>" << std::endl;
	*out << "<td >" << std::endl;
	*out << "Endpaths State" << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td" << (sched_->inhibit_endpaths_ ? " bgcolor=\"red\">" : ">") << std::endl;
	*out <<  (sched_->inhibit_endpaths_ ? "disabled" : "enabled") << std::endl;
	*out << "</td>" << std::endl;
	*out << "  </tr>"                                            << endl;
	*out << "<tr>" << std::endl;
	*out << "<td >" << std::endl;
	*out << "Global Input Prescale" << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td" << (sched_->global_input_prescale_!=1 ? " bgcolor=\"red\">" : ">") << std::endl;
	*out <<  sched_->global_input_prescale_ << std::endl;
	*out << "</td>" << std::endl;
	*out << "  </tr>"                                            << endl;
	*out << "<tr>" << std::endl;
	*out << "<td >" << std::endl;
	*out << "Global Output Prescale" << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td" << (sched_->global_output_prescale_!=1 ? " bgcolor=\"red\">" : ">") << std::endl;
	*out <<  sched_->global_output_prescale_ << std::endl;
	*out << "</td>" << std::endl;
	*out << "  </tr>"                                            << endl;


    
  *out << "</table>" << std::endl;

  *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">" << std::endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=3>"                                       << endl;
    *out << "      " << "Application"                                  << endl;

    if(descs_.size()>0)
      *out << " (Process name=" << descs_[0].processName_ << ")"       << endl;



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
	if(mwr && mwr->checkWeb(descs_[idesc].moduleName_))
	  *out << "<a href=\"/" << urn << "/moduleWeb?module=" << descs_[idesc].moduleName_ << "\">" 
	       << descs_[idesc].moduleName_ << "</a>" << std::endl;
	else
	  *out << descs_[idesc].moduleName_ << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td >" << std::endl;
	*out << descs_[idesc].moduleLabel_ << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td >" << std::endl;
	*out << descs_[idesc].versionNumber_ << std::endl;
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

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EventFilter/Utilities/interface/ModuleWebRegistry.h"

void EventProcessor::moduleWebPage(xgi::Input *in, xgi::Output *out, 
				   const std::string &mod)
{
  edm::ServiceRegistry::Operate operate(serviceToken_);
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

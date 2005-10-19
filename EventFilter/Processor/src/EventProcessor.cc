#include "EventFilter/Processor/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"
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
      edm::ParameterSet main_input = params_.getParameter<edm::ParameterSet>("@main_input");
      edm::InputSourceDescription isdesc(pname,pass,preg);
      boost::shared_ptr<edm::InputSource> input_
	(edm::InputSourceFactory::get()->makeInputSource(main_input, isdesc).release());
      return input_;
    } catch(const edm::Exception& iException) {
      if(edm::errors::Configuration == iException.categoryCode()) {
	throw edm::Exception(edm::errors::Configuration, "NoSource")
          <<"No main input source found in configuration.  Please add an input source via 'source = ...' in the configuration file.\n";
      } else {
	throw;
      }
    }
    return boost::shared_ptr<edm::InputSource>();
  }
  

  
}
using namespace evf;

EventProcessor::EventProcessor(int tid/*, boost::shared_ptr<edm::InputService> is*/) : 
  Task("FPTq"), /*input_(is),*/ tid_(tid),  serviceToken_(edm::ServiceToken()), 
  emittedBeginJob_(false), running_(false), paused_(false), eventcount(0)
{
  cout << "EventProcessor constructor " << endl;
}

EventProcessor::~EventProcessor()
{
  cout << "EventProcessor destructor " << endl;

}

void EventProcessor::init(std::string &config) 
{
  cout<<"FPEventProcessor::init()"<<endl;
  edm::ServiceToken iToken;
  edm::serviceregistry::ServiceLegacy iLegacy = edm::serviceregistry::kOverlapIsError;
  edm::ProcessPSetBuilder builder(config);
  
  //create the services
  boost::shared_ptr< std::vector<edm::ParameterSet> > pServiceSets(builder.getServicesPSets());
  //NOTE: FIX WHEN POOL BUG FIXED
  // we force in the LoadAllDictionaries service in order to work around a bug in POOL
  {
    edm::ParameterSet ps;
    std::string type("LoadAllDictionaries");
    ps.addParameter("@service_type",type);
    pServiceSets->push_back( ps );
  }
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

  serviceToken_.connectTo(activityRegistry_);

  //add the ProductRegistry as a service ONLY for the construction phase
  boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::ConstProductRegistry> > 
    reg(new edm::serviceregistry::ServiceWrapper<edm::ConstProductRegistry>( 
								  std::auto_ptr<edm::ConstProductRegistry>(new edm::ConstProductRegistry(preg_))));
  edm::ServiceToken tempToken( edm::ServiceRegistry::createContaining(reg, serviceToken_, edm::serviceregistry::kOverlapIsError));
  
  //make the services available
  edm::ServiceRegistry::Operate operate(tempToken);

  params_ = builder.getProcessPSet();

  act_table_ = edm::ActionTable(*params_);

  input_= makeInput(*params_, (*params_).getParameter<string>("@process_name"),
		    getPass(), preg_);
  edm::ScheduleBuilder sbuilder = 
    edm::ScheduleBuilder(*params_, wreg_, preg_, act_table_);
  workers_= (sbuilder.getPathList());
  runner_ = std::auto_ptr<edm::ScheduleExecutor>(new edm::ScheduleExecutor(workers_,act_table_));
  runner_->preModuleSignal.connect(activityRegistry_.preModuleSignal_);
  runner_->postModuleSignal.connect(activityRegistry_.postModuleSignal_);
  //  fillEventSetupProvider(esp_, *params_, common_);  
  using namespace std;
  using namespace edm::eventsetup;
  vector<string> providers = (*params_).getParameter<vector<string> >("@all_esmodules");
  for(vector<string>::iterator itName = providers.begin();
      itName != providers.end();
      ++itName) {
    edm::ParameterSet providerPSet = (*params_).getParameter<edm::ParameterSet>(*itName);
    ModuleFactory::get()->addTo(esp_, 
				providerPSet, 
				(*params_).getParameter<string>("@process_name"), 
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
  std::cout << "EventProcessor beginRun() " << std::endl;
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken_);
  running_=true;
  if(! emittedBeginJob_) {
    edm::EventSetup const& es = esp_.eventSetupForInstance(edm::IOVSyncValue::beginOfTime());
    PathList::iterator itWorkerList = workers_.begin();
    PathList::iterator itEnd = workers_.end();
    ESRefWrapper wrapper(es);
    
    for(; itWorkerList != itEnd; ++itEnd) {
      std::for_each(itWorkerList->begin(), itWorkerList->end(), 
		    boost::bind(boost::mem_fn(&edm::Worker::beginJob), _1, wrapper));
	for(WorkerList::const_iterator itWorker = itWorkerList->begin();
	    itWorker != itWorkerList->end();
	    ++itWorker) {
	  descs_.push_back((*itWorker)->description());
	}
    }
    emittedBeginJob_ = true;
    activityRegistry_.postBeginJobSignal_();
  }
}

bool EventProcessor::endRun() 
{
  bool returnValue = true;
  std::cout << "EventProcessor endRun() " << std::endl;
  if(running_)
    {    
      returnValue = false;
    }
  else
    {
      //make the services available
      edm::ServiceRegistry::Operate operate(serviceToken_);
      
      
      PathList::const_iterator itWorkerList = workers_.begin();
      PathList::const_iterator itEnd = workers_.end();
      for(; itWorkerList != itEnd; ++itEnd) {
	for(WorkerList::const_iterator itWorker = itWorkerList->begin();
	    itWorker != itWorkerList->end();
	    ++itWorker) {
	  try {
	    (*itWorker)->endJob();
	  } catch(cms::Exception& iException) {
	    cerr<<"Caught cms::Exception in endJob: "<< iException.what()<<endl;
	    returnValue = false;
	  } catch(std::exception& iException) {
	    cerr<<"Caught std::exception in endJob: "<< iException.what()<<endl;
	    cerr<<endl;
	    returnValue = false;
	  } catch(...) {
	    cerr<<"Caught unknown exception in endJob."<<endl;
	    returnValue = false;
	  }
	}
      }
      
    }     
  
  activityRegistry_.postEndJobSignal_();
  descs_.clear();
  return returnValue;
}
  


/**
 */                                             
#include "FWCore/Framework/interface/Event.h"

void EventProcessor::run()
{
  std::cout << "EventProcessor run() " << std::endl;
  //make the services available
  edm::ServiceRegistry::Operate operate(serviceToken_);

  while(running_)
    {

      if(paused_) 
	{
	  cout << "checking if event processor must be paused " << endl;
	  pause();
	  cout << "continuing with next event " << endl;
	}
      ++eventcount;
      auto_ptr<edm::EventPrincipal> pep = input_->readEvent();
      
      if(pep.get()==0) break;
      edm::IOVSyncValue ts(pep->id(), pep->time());
      edm::EventSetup const& es = esp_.eventSetupForInstance(ts);
      
      try
	{
	  edm::ModuleDescription dummy;
	  {
	    activityRegistry_.preProcessEventSignal_(pep->id(),pep->time());
	  }
	  edm::EventRegistry::Operate oper(pep->id(),pep.get());
	  runner_->runOneEvent(*pep.get(),es);
	  {
	    activityRegistry_.postProcessEventSignal_(edm::Event(*pep.get(),dummy) , es);
	  }
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
    }
}

#include "extern/cgicc/linuxx86/include/cgicc/CgiDefs.h"
#include "extern/cgicc/linuxx86/include/cgicc/Cgicc.h"
#include "extern/cgicc/linuxx86/include/cgicc/HTTPHTMLHeader.h"
#include "extern/cgicc/linuxx86/include/cgicc/HTMLClasses.h"

void EventProcessor::taskWebPage(xgi::Input *in, xgi::Output *out)
{

  evf::filter *filt = 0;

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
  *out << "</table>" << std::endl;

  *out << "<table frame=\"void\" rules=\"rows\" class=\"modules\">" << std::endl;
    *out << "  <tr>"                                                   << endl;
    *out << "    <th colspan=3>"                                       << endl;
    *out << "      " << "Application"                                  << endl;
    *out << "    </th>"                                                << endl;
    *out << "  </tr>"                                                  << endl;

        *out << "<tr >" << std::endl;
	*out << "<th >" << std::endl;
	*out << "Module" << std::endl;
	*out << "</th>" << std::endl;
	*out << "<th >" << std::endl;
	*out << "Version" << std::endl;
	*out << "</th>" << std::endl;
	*out << "<th >" << std::endl;
	*out << "Process" << std::endl;
	*out << "</th>" << std::endl;
	*out << "</tr>" << std::endl;

  for(unsigned int idesc = 0; idesc < descs_.size(); idesc++)
    {
        *out << "<tr>" << std::endl;
	*out << "<td >" << std::endl;
	*out << descs_[idesc].moduleName_ << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td >" << std::endl;
	*out << descs_[idesc].versionNumber_ << std::endl;
	*out << "</td>" << std::endl;
	*out << "<td >" << std::endl;
	*out << descs_[idesc].processName_ << std::endl;
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


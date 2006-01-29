
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/src/ProducerWorker.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/Log.h"

// needed for type tests
#include "FWCore/Framework/src/OutputWorker.h"
#include "FWCore/Framework/src/FilterWorker.h"

#include "boost/shared_ptr.hpp"
#include "boost/lambda/lambda.hpp"
#include "boost/lambda/algorithm.hpp"
#include "boost/bind.hpp"

#include <string>
#include <memory>
#include <vector>
#include <iomanip>
#include <list>

using namespace std;

namespace edm
{
  typedef TriggerResults::BitMask BitMask;
  typedef boost::shared_ptr<BitMask> BitMaskPtr;
  typedef boost::shared_ptr<Worker> WorkerPtr;
  typedef vector<string> vstring;

  namespace
  {
    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    WorkerPtr makeInserter(const ParameterSet& pset,
			   ProductRegistry& preg,
			   ActionTable& actions,
			   BitMaskPtr bm)
    {
#if 1
      ParameterSet trig_pset(pset.getUntrackedParameter<ParameterSet>("@trigger_paths"));
      string proc_name = pset.getParameter<string>("@process_name");
      WorkerParams work_args(trig_pset,preg,actions,proc_name);

      ModuleDescription md;
      md.pid = trig_pset.id();
      md.moduleName_ = "TriggerResultInserter";
      md.moduleLabel_ = "TriggerResults";
      md.processName_ = proc_name;
      md.versionNumber_ = 0; // not set properly!!!
      md.pass = 0; // not set properly!!!

      auto_ptr<EDProducer> producer(new TriggerResultInserter(trig_pset,bm));

      WorkerPtr ptr(new ProducerWorker(producer,md,work_args));
#else
      WorkerPtr ptr;
#endif
      return ptr;
    }

    class CallPrePost
    {
    public:
      CallPrePost(ActivityRegistry* a,
		  EventPrincipal* ep, 
		  const EventSetup* es):
	a_(a),ep_(ep),es_(es)
      { a_->preProcessEventSignal_(ep_->id(),ep_->time()); }
      ~CallPrePost()
      { 
	ModuleDescription dummy;
	Event evt(*ep_,dummy);
	const Event& eref(evt);
	a_->postProcessEventSignal_(eref,*es_);
      }

    private:
	ActivityRegistry* a_;
	EventPrincipal* ep_;
	const EventSetup* es_;
    };

  }

  Schedule::~Schedule() { }

  Schedule::Schedule(ParameterSet const& proc_pset,
		     WorkerRegistry& wreg,
		     ProductRegistry& preg,
		     ActionTable& actions,
		     boost::shared_ptr<ActivityRegistry> areg):
    pset_(proc_pset),
    worker_reg_(&wreg),
    prod_reg_(&preg),
    act_table_(&actions),
    proc_name_(proc_pset.getParameter<string>("@process_name")),
    trig_pset_(proc_pset.getUntrackedParameter<ParameterSet>("@trigger_paths",ParameterSet())),
    act_reg_(areg),
    state_(Ready),
    path_name_list_(trig_pset_.getParameter<vstring>("@paths")),
    end_path_name_list_(trig_pset_.getParameter<vstring>("@end_paths")),
    results_(new BitMask(path_name_list_.size())),
    results_inserter_(),
    trig_paths_(),
    end_paths_(),
    wantSummary_(false),
    makeTriggerResults_(false),
    total_events_(),
    total_passed_()
  {
    try
      {
        ParameterSet opts = pset_.getParameter<ParameterSet>("options");
	wantSummary_ = opts.getUntrackedParameter("wantSummary",false);
	makeTriggerResults_ = opts.getUntrackedParameter("makeTriggerResults",false);
      }
    catch(edm::Exception& e)
      {
      }

    vstring& trigs = path_name_list_;
    vstring& ends = end_path_name_list_;
    bool hasFilter = false;
    
    vstring::iterator ib(trigs.begin()),ie(trigs.end());
    for(int bitpos=0;ib!=ie;++ib,++bitpos)
      {
	hasFilter = fillTrigPath(bitpos,*ib);
      }

    // the results inserter is first in the end path
    if(hasFilter || makeTriggerResults_)
      {
    results_inserter_=makeInserter(proc_pset,preg,actions,results_);
	end_paths_.insert(end_paths_.begin(),results_inserter_.get());
	all_workers_.insert(results_inserter_.get());
      }

    vstring::iterator eib(ends.begin()),eie(ends.end());
    for(;eib!=eie;++eib)
      {
	fillEndPath(*eib);
      }

    prod_reg_->setProductIDs();
  }

  void Schedule::fillWorkers(const std::string& name, Workers& out)
  {
    vstring modnames = pset_.getParameter<vstring>(name);
    vstring::iterator it(modnames.begin()),ie(modnames.end());
    Workers tmpworkers;

    for(;it!=ie;++it)
      {
	ParameterSet modpset = pset_.getParameter<ParameterSet>(*it);
	unsigned long version=1, pass=1;
	WorkerParams params(modpset, *prod_reg_, *act_table_,
                            proc_name_, version, pass);
	Worker* w = worker_reg_->getWorker(params);
	tmpworkers.push_back(w);
      }

    out.swap(tmpworkers);
  }

  bool Schedule::fillTrigPath(int bitpos,const string& name)
  {
    Workers tmpworkers;
    Workers goodworkers;
    fillWorkers(name,tmpworkers);
    bool hasFilter = false;

    // check for any OutputModules
    for(Workers::iterator wi(tmpworkers.begin()),
	  we(tmpworkers.end());wi!=we;++wi)
      {
	if(dynamic_cast<OutputWorker*>(*wi)!=0)
	  {
	    LogWarning("path") << "OutputModule " 
			       << (*wi)->description().moduleLabel_
			       << " appears in path " << name << ".\n"
			       << "This will not be allowed in future releases.\n"
			       << "This module has been moved to the endpath.\n";

	    end_paths_.push_back(*wi);
	  }
	else
	  goodworkers.push_back(*wi);

	if(dynamic_cast<FilterWorker*>(*wi)!=0)
	  hasFilter = true;
      }

    Path p(bitpos,name,goodworkers,results_,pset_,*act_table_,act_reg_);
    trig_paths_.push_back(p);
    all_workers_.insert(tmpworkers.begin(),tmpworkers.end());

    return hasFilter;
  }

  void Schedule::fillEndPath(const string& name)
  {
    Workers tmpworkers;
    fillWorkers(name,tmpworkers);
    end_paths_.insert(end_paths_.end(),tmpworkers.begin(),tmpworkers.end());
    all_workers_.insert(tmpworkers.begin(),tmpworkers.end());
  }

  void Schedule::runOneEvent(EventPrincipal& ep, EventSetup const& es)
  {
    resetWorkers();
    results_->reset();
    state_ = Running;
    ++total_events_;

    try
      {
	CallPrePost cpp(act_reg_.get(),&ep,&es);

	// go through triggering paths first
	bool result = false;
	int which_one = 0;
	TrigPaths::iterator ti(trig_paths_.begin()),te(trig_paths_.end());
	for(;ti!=te;++ti)
	  {
	    ti->runOneEvent(ep,es);
	    result += (*results_)[which_one];
	    ++which_one;
	  }

	if(result) ++total_passed_;
	state_ = Latched;
	
	// go through end paths next
	NonTrigPaths::iterator ei(end_paths_.begin()),ee(end_paths_.end());
	for(;ei!=ee;++ei)
	  {
	    (*ei)->doWork(ep,es);
	  }
      }
    catch(cms::Exception& e)
      {
	actions::ActionCodes code = act_table_->find(e.rootCause());

	switch(code)
	  {
	  case actions::IgnoreCompletely:
	    {
	      LogWarning(e.category())
		<< "exception being ignored for current event:\n"
		<< e.what();
	      break;
	    }
	  case actions::SkipEvent:
	    {
	      LogWarning(e.category())
		<< "an exception occurred and event is being skipped: \n"
		<< e.what();
	      break;
	    }
	  default:
	    {
	      LogError(e.category())
		<< "an exception ocurred during current event processing\n";
	      state_ = Ready;
	      throw edm::Exception(errors::EventProcessorFailure,
				   "EventProcessingStopped",e)
		<< "an exception ocurred during current event processing\n";
	    }
	  }
      }
    catch(...)
      {
	LogError("PassingThrough")
	  << "an exception ocurred during current event processing\n";
	state_ = Ready;
	throw;
      }

    // next thing probably is not needed, the product insertion code clears it
    state_ = Ready;
  }

  using namespace boost::lambda;

  void Schedule::endJob()
  {
    AllWorkers::iterator i(all_workers_.begin()),e(all_workers_.end());
    for(;i!=e;++i) { (*i)->endJob(); }

    //for_each(all_workers_.begin(),all_workers_.end(),
    //		    boost::bind(&Worker::endJob,_1));    

    if(wantSummary_ == false) return;

    cout << "trigreport " << "---------- Path Summary ----------\n";
    cout << "trigreport "
	 << right << setw(4)  << "Bit" << " "
	 << right << setw(10) << "Passed" << " "
	 << right << setw(10) << "Failed" << " "
	 << "Name" << "\n";

    TrigPaths::iterator pi(trig_paths_.begin()),pe(trig_paths_.end());
    for(;pi!=pe;++pi)
      {
	cout << "trigreport "
	     << right << setw(4)  << pi->bitPosition() << " "
	     << right << setw(10) << pi->timesPassed() << " "
	     << right << setw(10) << (pi->timesVisited() - pi->timesPassed()) << " "
	     << pi->name() << "\n";
      }

    cout << "trigreport " << "---------- Module Summary ----------\n";
    cout << "trigreport "
	 << right << setw(10) << "Passed" << " "
	 << right << setw(10) << "Failed" << " "
	 << right << setw(10) << "Run" << " "
	 << right << setw(10) << "Visited" << " "
	 << right << setw(10) << "Error" << " "
	 << "Name" << "\n";

    AllWorkers::iterator ai(all_workers_.begin()),ae(all_workers_.end());
    for(;ai!=ae;++ai)
      {
	cout << "trigreport "
	     << right << setw(10) << (*ai)->timesPass() << " "
	     << right << setw(10) << (*ai)->timesFailed() << " "
	     << right << setw(10) << (*ai)->timesRun() << " "
	     << right << setw(10) << (*ai)->timesVisited() << " "
	     << right << setw(10) << (*ai)->timesExcept() << " "
	     << (*ai)->description().moduleLabel_ << "\n";
	  
      }

    cout << "trigreport " << "---------- Event Summary ------------\n";
    cout << "trigreport"
	 << " Event total = " << total_events_
	 << " Passed = " << total_passed_
	 << " Failed = " << (total_events_ - total_passed_)
	 << "\n";

    // the module-in-path stats are not reported here and could be!
  }

  void Schedule::beginJob(EventSetup const& es)
  {
    AllWorkers::iterator i(all_workers_.begin()),e(all_workers_.end());
    for(;i!=e;++i) { (*i)->beginJob(es); }

    //for_each(all_workers_.begin(),all_workers_.end(),
    //		    boost::bind(&Worker::beginJob,_1,es));    
  }

  void Schedule::resetWorkers()
  {
    AllWorkers::iterator i(all_workers_.begin()),e(all_workers_.end());
    for(;i!=e;++i) { (*i)->reset(); }

    //for_each(all_workers_.begin(),all_workers_.end(),
    //		    boost::bind(&Worker::reset,_1));
  }

  
}

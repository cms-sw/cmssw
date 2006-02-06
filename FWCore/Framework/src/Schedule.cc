
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/src/ProducerWorker.h"
#include "FWCore/Framework/src/WorkerInPath.h"
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
#include <algorithm>

using namespace std;

namespace edm
{
  namespace
  {
    // -----------------------------

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    Schedule::WorkerPtr makeInserter(const ParameterSet& trig_pset,
				     const string& proc_name,
				     ProductRegistry& preg,
				     ActionTable& actions,
				     Schedule::BitMaskPtr bm)
    {
#if 1
      WorkerParams work_args(trig_pset,preg,actions,proc_name);
      ModuleDescription md;
      md.pid = trig_pset.id();
      md.moduleName_ = "TriggerResultInserter";
      md.moduleLabel_ = "TriggerResults";
      md.processName_ = proc_name;
      md.versionNumber_ = 0; // not set properly!!!
      md.pass = 0; // not set properly!!!

      auto_ptr<EDProducer> producer(new TriggerResultInserter(trig_pset,bm));

      Schedule::WorkerPtr ptr(new ProducerWorker(producer,md,work_args));
#else
      Schedule::WorkerPtr ptr;
#endif
      return ptr;
    }

  // -----------------------------

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

  // -----------------------------

  typedef vector<string> vstring;

  void checkIfSubset(const vstring& in_all, const vstring& in_sub)
  {
    vstring all(in_all), sub(in_sub), result;
    sort(all.begin(),all.end());
    sort(sub.begin(),sub.end());
    set_intersection(all.begin(),all.end(),
		     sub.begin(),sub.end(),
		     back_inserter(result));

    if(result.size() != sub.size())
      throw cms::Exception("TriggerPaths")
	<< "Specified listOfTriggers is not a subset of the available paths\n";
  }

  ParameterSet getTrigPSet(ParameterSet const& proc_pset)
  {
    ParameterSet rc = 
      proc_pset.getUntrackedParameter<ParameterSet>("@trigger_paths");
    bool want_results = false;
    // default for trigger paths is all the paths
    vstring allpaths = rc.getParameter<vstring>("@paths");

    // the value depends on options and value of listOfTriggers
    try
      {
        ParameterSet opts = proc_pset.getParameter<ParameterSet>("options");
	want_results =
	  opts.getUntrackedParameter<bool>("makeTriggerResults",false);

	// if makeTriggerResults is true, then listOfTriggers must be given

	if(want_results)
	  {
	    vstring tmppaths = opts.getParameter<vstring>("listOfTriggers");

	    // verify that all then names in allpaths are a subset of
	    // the names currently in allpaths (all the names)

	    if(!tmppaths.empty() && tmppaths[0] == "*")
	      {
		// leave as full list
	      }
	    else
	      {
		checkIfSubset(allpaths, tmppaths);
		allpaths.swap(tmppaths);
	      }
	  }
      }
    catch(edm::Exception& e)
      {
      }

    rc.addParameter<vstring>("@trigger_paths",allpaths);
    return rc;
  }

  // -----------------------------

  Schedule::~Schedule() { }

  Schedule::Schedule(ParameterSet const& proc_pset,
		     WorkerRegistry& wreg,
		     ProductRegistry& preg,
		     ActionTable& actions,
		     ActivityRegistryPtr areg):
    pset_(proc_pset),
    worker_reg_(&wreg),
    prod_reg_(&preg),
    act_table_(&actions),
    proc_name_(proc_pset.getParameter<string>("@process_name")),
    trig_pset_(getTrigPSet(proc_pset)),
    act_reg_(areg),
    state_(Ready),
    trig_name_list_(trig_pset_.getParameter<vstring>("@trigger_paths")),
    path_name_list_(trig_pset_.getParameter<vstring>("@paths")),
    end_path_name_list_(trig_pset_.getParameter<vstring>("@end_paths")),
    trig_name_set_(trig_name_list_.begin(),trig_name_list_.end()),

    results_(new BitMask),
    results_bit_count_(trig_name_list_.size()),
    nontrig_results_(new BitMask),
    nontrig_results_bit_count_(path_name_list_.size()),
    endpath_results_(new BitMask),
    // extra position in endpath_results is for wrongly-placed modules
    endpath_results_bit_count_(end_path_name_list_.size()+1),

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
    catch(edm::Exception& e) { }

    vstring& ends = end_path_name_list_;
    bool hasFilter = false;
    
    vstring::iterator ib(path_name_list_.begin()),ie(path_name_list_.end());
    int trig_bitpos=0, non_bitpos=0;

    for(;ib!=ie;++ib)
      {
	if(trig_name_set_.find(*ib)!=trig_name_set_.end())
	  {
	    hasFilter += fillTrigPath(trig_bitpos,*ib, results_);
	    ++trig_bitpos;
	  }
	else
	  {
	    fillTrigPath(non_bitpos,*ib, nontrig_results_);
	    ++non_bitpos;
	  }
      }
    
    // the results inserter stands alone
    if(hasFilter || makeTriggerResults_)
      {
	results_inserter_=makeInserter(trig_pset_,proc_name_,
				       preg,actions,results_);
	all_workers_.insert(results_inserter_.get());
      }

    handleWronglyPlacedModules();

    vstring::iterator eib(ends.begin()),eie(ends.end());
    for(int bitpos=0;eib!=eie;++eib,++bitpos)
      {
	fillEndPath(bitpos,*eib);
      }

    prod_reg_->setProductIDs();
  }

  void Schedule::handleWronglyPlacedModules()
  {
    // the wrongly placed workers (always output modules)
    // are already accounted for, but are not yet in paths.
    // Here we do that path assignment.

    if(!tmp_wrongly_placed_.empty())
      {
	unsigned int pos = endpath_results_bit_count_-1;
	Path p(pos,"WronglyPlaced",tmp_wrongly_placed_,
	       endpath_results_,pset_,*act_table_,act_reg_);
    end_paths_.push_back(p);
      }
  }


  void Schedule::fillWorkers(const std::string& name, PathWorkers& out)
  {
    vstring modnames = pset_.getParameter<vstring>(name);
    vstring::iterator it(modnames.begin()),ie(modnames.end());
    PathWorkers tmpworkers;

    for(;it!=ie;++it)
      {
	bool invert = (*it)[0]=='!';
	string realname = invert?string(it->begin()+1,it->end()):*it;
	WorkerInPath::State state =
	  invert ? WorkerInPath::Veto : WorkerInPath::Normal;

	ParameterSet modpset = pset_.getParameter<ParameterSet>(realname);
	unsigned long version=1, pass=1;
	WorkerParams params(modpset, *prod_reg_, *act_table_,
                            proc_name_, version, pass);
	WorkerInPath w(worker_reg_->getWorker(params),state);
	tmpworkers.push_back(w);
      }

    out.swap(tmpworkers);
  }

  struct ToWorker
  {
    Worker* operator()(WorkerInPath& w) const { return w.getWorker(); }
  };

  bool Schedule::fillTrigPath(int bitpos,const string& name, BitMaskPtr ptr)
  {
    PathWorkers tmpworkers;
    PathWorkers goodworkers;
    Workers holder;
    fillWorkers(name,tmpworkers);
    bool hasFilter = false;

    // check for any OutputModules
    for(PathWorkers::iterator wi(tmpworkers.begin()),
	  we(tmpworkers.end());wi!=we;++wi)
      {
	Worker* tworker = wi->getWorker();
	if(dynamic_cast<OutputWorker*>(tworker)!=0)
	  {
	    LogWarning("path")
	      << "OutputModule " 
	      << tworker->description().moduleLabel_
	      << " appears in path " << name << ".\n"
	      << "This will not be allowed in future releases.\n"
	      << "This module has been moved to the endpath.\n";

	    tmp_wrongly_placed_.push_back(*wi);
	  }
	else
	  goodworkers.push_back(*wi);

	if(dynamic_cast<FilterWorker*>(tworker)!=0)
	  hasFilter = true;

	holder.push_back(tworker);
      }

    // an empty path will cause an extra bit that is not used
    if(!goodworkers.empty())
	{
        Path p(bitpos,name,goodworkers,ptr,pset_,*act_table_,act_reg_);
        trig_paths_.push_back(p);
	}
    all_workers_.insert(holder.begin(),holder.end());

    return hasFilter;
  }

  void Schedule::fillEndPath(int bitpos,const string& name)
  {
    PathWorkers tmpworkers;
    fillWorkers(name,tmpworkers);
    Workers holder;

    transform(tmpworkers.begin(),tmpworkers.end(),
	      back_inserter(holder),ToWorker());
    
    Path p(bitpos,name,tmpworkers,endpath_results_,pset_,*act_table_,act_reg_);
    end_paths_.push_back(p);
    all_workers_.insert(holder.begin(),holder.end());
  }

  void Schedule::runOneEvent(EventPrincipal& ep, EventSetup const& es)
  {
    resetWorkers();
    results_->reset();
    endpath_results_->reset();
    state_ = Running;
    ++total_events_;

    try
      {
	CallPrePost cpp(act_reg_.get(),&ep,&es);

	// go through triggering paths first
	bool result = false;

	TrigPaths::iterator ti(trig_paths_.begin()),te(trig_paths_.end());
	for(int which_one=0;ti!=te;++ti,++which_one)
	  {
	    ti->runOneEvent(ep,es);
	    result += (*results_)[which_one];
	  }

	if(result) ++total_passed_;
	state_ = Latched;

	if(results_inserter_.get()) results_inserter_->doWork(ep,es);
	
	// go through end paths next
	TrigPaths::iterator ei(end_paths_.begin()),ee(end_paths_.end());
	for(;ei!=ee;++ei)
	  {
	    ei->runOneEvent(ep,es);
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

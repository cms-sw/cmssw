
#include "FWCore/Framework/interface/Schedule.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/Framework/interface/TriggerReport.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/src/ProducerWorker.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/WorkerRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/PassID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"
#include "FWCore/Framework/src/TriggerResultInserter.h"


// needed for type tests
#include "FWCore/Framework/src/OutputWorker.h"
#include "FWCore/Framework/src/FilterWorker.h"

#include "boost/shared_ptr.hpp"
#include "boost/bind.hpp"
#include "boost/lambda/lambda.hpp"
#include "boost/lambda/bind.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <vector>


namespace edm {
  namespace {

    // Function template to transform each element in the input range to
    // a value placed into the output range. The supplied function
    // should take a const_reference to the 'input', and write to a
    // reference to the 'output'.
    template <class InputIterator, class ForwardIterator, class Func>
    void
    transform_into(InputIterator begin, InputIterator end,
		   ForwardIterator out, Func func) {
      for (; begin != end; ++begin, ++out) func(*begin, *out);
    }
    
    // Function template that takes a sequence 'from', a sequence
    // 'to', and a callable object 'func'. It and applies
    // transform_into to fill the 'to' sequence with the values
    // calcuated by the callable object, taking care to fill the
    // outupt only if all calls succeed.
    template <class FROM, class TO, class FUNC>
    void
    fill_summary(FROM const& from, TO& to, FUNC func) {
      TO temp(from.size());
      transform_into(from.begin(), from.end(), temp.begin(), func);
      to.swap(temp);
    }

    // -----------------------------

    // Here we make the trigger results inserter directly.  This should
    // probably be a utility in the WorkerRegistry or elsewhere.

    Schedule::WorkerPtr 
    makeInserter(ParameterSet const& proc_pset,
		 ParameterSet const& trig_pset,
		 std::string const& proc_name,
		 ProductRegistry& preg,
		 ActionTable& actions,
		 Schedule::TrigResPtr trptr) {
#if 1
      WorkerParams work_args(proc_pset,trig_pset,preg,actions,proc_name);
      ModuleDescription md;
      md.parameterSetID_ = trig_pset.id();
      md.moduleName_ = "TriggerResultInserter";
      md.moduleLabel_ = "TriggerResults";
      md.processConfiguration_ = ProcessConfiguration(proc_name, proc_pset.id(), getReleaseVersion(), getPassID());

      std::auto_ptr<EDProducer> producer(new TriggerResultInserter(trig_pset,trptr));

      Schedule::WorkerPtr ptr(new ProducerWorker(producer,md,work_args));
#else
      Schedule::WorkerPtr ptr;
#endif
      return ptr;
    }

    // -----------------------------

      Schedule::CallPrePost::CallPrePost(ActivityRegistry* a,
		  EventPrincipal* ep,
		  EventSetup const* es) :
	a_(a),ep_(ep),es_(es) {
	// Avoid possible order-of-evaluation trouble
	// by not having two function calls as actual arguments.
	EventID id = ep_->id();
	a_->preProcessEventSignal_(id, ep_->time()); 
      }

      Schedule::CallPrePost::~CallPrePost() {
        if (ep_) {
          Event ev(*ep_, ModuleDescription());
	  a_->postProcessEventSignal_(ev, *es_);
        }
      }

  // -----------------------------

  typedef std::vector<std::string> vstring;

  // -----------------------------

  Schedule::Schedule(ParameterSet const& proc_pset,
		     edm::service::TriggerNamesService& tns,
		     WorkerRegistry& wreg,
		     ProductRegistry& preg,
		     ActionTable& actions,
		     ActivityRegistryPtr areg):
    pset_(proc_pset),
    worker_reg_(&wreg),
    prod_reg_(&preg),
    act_table_(&actions),
    processName_(tns.getProcessName()),
    trig_pset_(tns.getTrigPSet()),
    act_reg_(areg),
    state_(Ready),
    trig_name_list_(tns.getTrigPaths()),
    path_name_list_(tns.getPaths()),
    end_path_name_list_(tns.getEndPaths()),
    trig_name_set_(trig_name_list_.begin(),trig_name_list_.end()),

    results_        (new HLTGlobalStatus(trig_name_list_.size())),
    nontrig_results_(new HLTGlobalStatus(path_name_list_.size())),
    endpath_results_(), // delay!
    results_inserter_(),
    all_workers_(),
    all_output_workers_(),
    trig_paths_(),
    end_paths_(),
    tmp_wrongly_placed_(),
    wantSummary_(tns.wantSummary()),
    makeTriggerResults_(tns.makeTriggerResults()),
    total_events_(),
    total_passed_(),
    stopwatch_(new RunStopwatch::StopwatchPointer::element_type),
    unscheduled_(new UnscheduledCallProducer),
    demandBranches_(),
    endpathsAreActive_(true)
  {
    ParameterSet maxEventsPSet(pset_.getUntrackedParameter<ParameterSet>("maxEvents", ParameterSet()));

    std::string const input("input");
    std::string const output("output");

    int maxEventSpecs = 0; 
    int maxEventsIn = -1;
    int maxEventsOut = -1;
    ParameterSet vMaxEventsOut;
    std::vector<std::string> intNames = maxEventsPSet.getParameterNamesForType<int>(false);
    if (intNames.end() != std::find(intNames.begin(), intNames.end(), input)) {
      maxEventsIn = maxEventsPSet.getUntrackedParameter<int>(input);
      ++maxEventSpecs;
    }
    if (intNames.end() != std::find(intNames.begin(), intNames.end(), output)) {
      maxEventsOut = maxEventsPSet.getUntrackedParameter<int>(output);
      ++maxEventSpecs;
    }
    std::vector<std::string> psetNames;
    maxEventsPSet.getParameterSetNames(psetNames, false);
    if (psetNames.end() != std::find(psetNames.begin(), psetNames.end(), output)) {
      vMaxEventsOut = maxEventsPSet.getUntrackedParameter<ParameterSet>(output);
      ++maxEventSpecs;
    }

    if (maxEventSpecs > 1) {
        throw edm::Exception(edm::errors::Configuration) <<
	 "\nAt most, one of 'input' and 'output' may appear in the 'maxEvents' parameter set";
    }

    ParameterSet opts(pset_.getUntrackedParameter<ParameterSet>("options", ParameterSet()));

    bool hasFilter = false;
    int trig_bitpos=0, non_bitpos=0;
    std::set<std::string>::const_iterator trig_name_set_end = trig_name_set_.end();
    for (vstring::const_iterator i = path_name_list_.begin(),
	    e = path_name_list_.end();
	  i != e;
	  ++i) {
	 if (trig_name_set_.find(*i) != trig_name_set_end) {
	     hasFilter += fillTrigPath(trig_bitpos,*i, results_);
	     ++trig_bitpos;
	 } else {
	     fillTrigPath(non_bitpos,*i, nontrig_results_);
	     ++non_bitpos;
	 }
    }

    // the results inserter stands alone
    if(hasFilter || makeTriggerResults_) {
      results_inserter_=makeInserter(pset_,trig_pset_,processName_,
				     preg,actions,results_);
      all_workers_.insert(results_inserter_.get());
    }

    // check whether an endpath for wrongly placed modules is needed
    if(tmp_wrongly_placed_.empty()) {
      TrigResPtr epptr(new HLTGlobalStatus(end_path_name_list_.size()));
      endpath_results_ = epptr;
    } else {
      TrigResPtr epptr(new HLTGlobalStatus(end_path_name_list_.size()+1));
      endpath_results_ = epptr;
    }

    // fill normal endpaths
    vstring::iterator eib(end_path_name_list_.begin()),eie(end_path_name_list_.end());
    for(int bitpos = 0; eib != eie; ++eib, ++bitpos) fillEndPath(bitpos, *eib);

    // handle additional endpath containing wrongly placed modules
    handleWronglyPlacedModules();

    //See if all modules were used
    std::set<std::string> usedWorkerLabels;
    for(AllWorkers::iterator itWorker=workersBegin();
        itWorker != workersEnd();
        ++itWorker) {
      usedWorkerLabels.insert((*itWorker)->description().moduleLabel_);
    }
    std::vector<std::string> modulesInConfig(proc_pset.getParameter<std::vector<std::string> >("@all_modules"));
    std::set<std::string> modulesInConfigSet(modulesInConfig.begin(),modulesInConfig.end());
    std::vector<std::string> unusedLabels;
    set_difference(modulesInConfigSet.begin(),modulesInConfigSet.end(),
		   usedWorkerLabels.begin(),usedWorkerLabels.end(),
		   back_inserter(unusedLabels));
    //does the configuration say we should allow on demand?
    bool allowUnscheduled = opts.getUntrackedParameter<bool>("allowUnscheduled", false);
    std::set<std::string> unscheduledLabels;
    if(!unusedLabels.empty()) {
      //Need to
      // 1) create worker
      // 2) if they are ProducerWorkers, add them to our list
      // 3) hand list to our delayed reader
      std::vector<std::string>  shouldBeUsedLabels;
	
      for(std::vector<std::string>::iterator itLabel = unusedLabels.begin(), itLabelEnd = unusedLabels.end();
	  itLabel != itLabelEnd;
	  ++itLabel) {
	if (allowUnscheduled) {
	  unscheduledLabels.insert(*itLabel);
	  //Need to hold onto the parameters long enough to make the call to getWorker
	  ParameterSet workersParams(proc_pset.getParameter<ParameterSet>(*itLabel));
	  WorkerParams params(proc_pset, workersParams,
			      *prod_reg_, *act_table_,
			      processName_, getReleaseVersion(), getPassID());
	  Worker* newWorker(wreg.getWorker(params));
	  if (dynamic_cast<ProducerWorker*>(newWorker)) {
	    unscheduled_->addWorker(newWorker);
	    //add to list so it gets reset each new event
	    all_workers_.insert(newWorker);
	  } else {
	    //not a producer so should be marked as not used
	    shouldBeUsedLabels.push_back(*itLabel);
	  }
	} else {
	  //everthing is marked are unused so no 'on demand' allowed
	  shouldBeUsedLabels.push_back(*itLabel);
	}
      }
      if(!shouldBeUsedLabels.empty()) {
	std::ostringstream unusedStream;
	unusedStream << "'"<< shouldBeUsedLabels.front() <<"'";
	for(std::vector<std::string>::iterator itLabel = shouldBeUsedLabels.begin() + 1,
            itLabelEnd = shouldBeUsedLabels.end();
	    itLabel != itLabelEnd;
	    ++itLabel) {
	  unusedStream <<",'" << *itLabel<<"'";
	}
	LogWarning("path")
	  << "The following module labels are not assigned to any path:\n"
	  <<unusedStream.str()
	  <<"\n";
      }
    }
    prod_reg_->setProductIDs();

    // Set up all_output_workers_ if limiting the amount of output.
    if (maxEventsOut >= 0) {
      for (AllWorkers::const_iterator it = workersBegin(), itEnd = workersEnd();
	  it != itEnd; ++it) {
	OutputWorker const* workerPtr = dynamic_cast<OutputWorker*>(*it);
	if (workerPtr) {
          all_output_workers_.push_back(std::make_pair(maxEventsOut, workerPtr));
	}
      }
      if (all_output_workers_.empty()) {
	throw edm::Exception(edm::errors::Configuration) <<
	  "\nMaximum output specified, and there are no output modules configured.\n";
      }
    } else if (!vMaxEventsOut.empty()) {
      for (AllWorkers::const_iterator it = workersBegin(), itEnd = workersEnd();
	  it != itEnd; ++it) {
        OutputWorker const* workerPtr = dynamic_cast<OutputWorker*>(*it);
	if (workerPtr) {
	  std::string moduleLabel = workerPtr->description().moduleLabel_;
	  try {
            all_output_workers_.push_back(std::make_pair(vMaxEventsOut.getUntrackedParameter<int>(moduleLabel), workerPtr));
	  } catch (edm::Exception) {
	    throw edm::Exception(edm::errors::Configuration) <<
	      "\nNo entry in 'maxEvents' for output module label '" << moduleLabel << "'.\n";
	  }
	}
      }
    }

    //Now that these have been set, we can create the list of Branches we need for the 'on demand'
    ProductRegistry::ProductList const& prodsList = prod_reg_->productList();
    for(ProductRegistry::ProductList::const_iterator itProdInfo = prodsList.begin(),
        itProdInfoEnd = prodsList.end();
        itProdInfo != itProdInfoEnd;
        ++itProdInfo) {
	if(unscheduledLabels.end() != unscheduledLabels.find(itProdInfo->second.moduleLabel())) {
          boost::shared_ptr<Provenance> prov(new Provenance(itProdInfo->second, BranchEntryDescription::CreatorNotRun));
          demandBranches_.push_back(prov);
	}
      }
  }

  bool const Schedule::terminate() const {
    if (all_output_workers_.empty()) {
      // not terminating on output event count.
      return false;
    }
    for (AllOutputWorkers::const_iterator it = all_output_workers_.begin(),
	itEnd = all_output_workers_.end();
	it != itEnd; ++it) {
      if (it->first < 0 || it->second->eventCount() < it->first) {
        // Found an output module that has not reached output event count.
        return false;
      }
    }
    LogInfo("SuccessfulTermination")
      << "The job is terminating successfully because each output module\n"
      << "has reached its configured limit.\n";
    return true;
  }

  void Schedule::handleWronglyPlacedModules() {
    // the wrongly placed workers (always output modules)
    // are already accounted for, but are not yet in paths.
    // Here we do that path assignment.

    if(!tmp_wrongly_placed_.empty()) {
      std::string const newname("WronglyPlaced");
      unsigned int const pos(end_path_name_list_.size());
      Path p(pos,newname,tmp_wrongly_placed_,
	     endpath_results_,pset_,*act_table_,act_reg_);
      end_paths_.push_back(p);
      end_path_name_list_.push_back(newname);
    }
  }


  void Schedule::fillWorkers(std::string const& name, PathWorkers& out) {
    vstring modnames = pset_.getParameter<vstring>(name);
    vstring::iterator it(modnames.begin()),ie(modnames.end());
    PathWorkers tmpworkers;

    for(; it != ie; ++it) {

      WorkerInPath::FilterAction filterAction = WorkerInPath::Normal;
      if ((*it)[0] == '!')       filterAction = WorkerInPath::Veto;
      else if ((*it)[0] == '-')  filterAction = WorkerInPath::Ignore;

      std::string realname = *it;
      if (filterAction != WorkerInPath::Normal) realname.erase(0,1);

      ParameterSet modpset;
      try {
        modpset= pset_.getParameter<ParameterSet>(realname);
      } catch(cms::Exception&) {
        std::string pathType("endpath");
        if(find(end_path_name_list_.begin(),end_path_name_list_.end(), name) == end_path_name_list_.end()) {
          pathType = std::string("path");
        }
        throw edm::Exception(edm::errors::Configuration)<<"The unknown module label \""<<realname<<"\" appears in "<<pathType<<" \""<<name
							<<"\"\n please check spelling or remove that label from the path.";
      }
      WorkerParams params(pset_, modpset, *prod_reg_, *act_table_,
			  processName_, getReleaseVersion(), getPassID());
      WorkerInPath w(worker_reg_->getWorker(params), filterAction);
      tmpworkers.push_back(w);
    }

    out.swap(tmpworkers);
  }

  struct ToWorker {
    Worker* operator()(WorkerInPath& w) const { return w.getWorker(); }
  };

  bool Schedule::fillTrigPath(int bitpos, std::string const& name, TrigResPtr trptr) {
    PathWorkers tmpworkers;
    PathWorkers goodworkers;
    Workers holder;
    fillWorkers(name,tmpworkers);
    bool hasFilter = false;

    // check for any OutputModules
    for(PathWorkers::iterator wi(tmpworkers.begin()),
	  we(tmpworkers.end()); wi != we; ++wi) {
      Worker* tworker = wi->getWorker();
      if(dynamic_cast<OutputWorker*>(tworker)!=0) {
	LogWarning("path")
	  << "OutputModule "
	  << tworker->description().moduleLabel_
	  << " appears in path " << name << ".\n"
	  << "This will not be allowed in future releases.\n"
	  << "This module has been moved to the endpath.\n";

	tmp_wrongly_placed_.push_back(*wi);
      } else {
	goodworkers.push_back(*wi);
      }

      if(dynamic_cast<FilterWorker*>(tworker)!=0) {
	hasFilter = true;
      }

      holder.push_back(tworker);
    }

    // an empty path will cause an extra bit that is not used
    if(!goodworkers.empty()) {
      Path p(bitpos,name,goodworkers,trptr,pset_,*act_table_,act_reg_);
      trig_paths_.push_back(p);
    }
    all_workers_.insert(holder.begin(),holder.end());

    return hasFilter;
  }

  void Schedule::fillEndPath(int bitpos, std::string const& name) {
    PathWorkers tmpworkers;
    fillWorkers(name,tmpworkers);
    Workers holder;

    transform(tmpworkers.begin(),tmpworkers.end(),
	      back_inserter(holder),ToWorker());

    Path p(bitpos,name,tmpworkers,endpath_results_,pset_,*act_table_,act_reg_);
    end_paths_.push_back(p);
    all_workers_.insert(holder.begin(), holder.end());
  }

  void Schedule::endJob() {
    bool failure = false;
    cms::Exception accumulated("endJob");
    AllWorkers::iterator ai(workersBegin()),ae(workersEnd());
    for(; ai != ae; ++ai) {
      try {
	(*ai)->endJob();
      }
      catch (cms::Exception& e) {
        accumulated << "cms::Exception caught in Schedule::endJob\n"
		    << e.explainSelf();
        failure = true;
      }
      catch (std::exception& e) {
        accumulated << "Standard library exception caught in Schedule::endJob\n"
		    << e.what();
        failure = true;
      }
      catch (...) {
        accumulated << "Unknown exception caught in Schedule::endJob\n";
        failure = true;
      }
    }
    if (failure) {
      throw accumulated;
    }


    if(wantSummary_ == false) return;

    TrigPaths::const_iterator pi,pe;

    // The trigger report (pass/fail etc.):

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TrigReport " << "---------- Event  Summary ------------";
    LogVerbatim("FwkSummary") << "TrigReport"
	 << " Events total = " << totalEvents()
	 << " passed = " << totalEventsPassed()
	 << " failed = " << (totalEventsFailed())
	 << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TrigReport " << "---------- Path   Summary ------------";
    LogVerbatim("FwkSummary") << "TrigReport "
	 << std::right << std::setw(10) << "Trig Bit#" << " "
	 << std::right << std::setw(10) << "Run" << " "
	 << std::right << std::setw(10) << "Passed" << " "
	 << std::right << std::setw(10) << "Failed" << " "
	 << std::right << std::setw(10) << "Error" << " "
	 << "Name" << "";
    pi=trig_paths_.begin();
    pe=trig_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TrigReport "
	   << std::right << std::setw( 5) << (trig_name_set_.find(pi->name()) != trig_name_set_.end())
	   << std::right << std::setw( 5) << pi->bitPosition() << " "
	   << std::right << std::setw(10) << pi->timesRun() << " "
	   << std::right << std::setw(10) << pi->timesPassed() << " "
	   << std::right << std::setw(10) << pi->timesFailed() << " "
	   << std::right << std::setw(10) << pi->timesExcept() << " "
	   << pi->name() << "";
    }

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TrigReport " << "-------End-Path   Summary ------------";
    LogVerbatim("FwkSummary") << "TrigReport "
	 << std::right << std::setw(10) << "Trig Bit#" << " "
	 << std::right << std::setw(10) << "Run" << " "
	 << std::right << std::setw(10) << "Passed" << " "
	 << std::right << std::setw(10) << "Failed" << " "
	 << std::right << std::setw(10) << "Error" << " "
	 << "Name" << "";
    pi=end_paths_.begin();
    pe=end_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TrigReport "
	   << std::right << std::setw( 5) << (trig_name_set_.find(pi->name()) != trig_name_set_.end())
	   << std::right << std::setw( 5) << pi->bitPosition() << " "
	   << std::right << std::setw(10) << pi->timesRun() << " "
	   << std::right << std::setw(10) << pi->timesPassed() << " "
	   << std::right << std::setw(10) << pi->timesFailed() << " "
	   << std::right << std::setw(10) << pi->timesExcept() << " "
	   << pi->name() << "";
    }

    pi=trig_paths_.begin();
    pe=trig_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "---------- Modules in Path: " << pi->name() << " ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
	   << std::right << std::setw(10) << "Trig Bit#" << " "
	   << std::right << std::setw(10) << "Visited" << " "
	   << std::right << std::setw(10) << "Passed" << " "
	   << std::right << std::setw(10) << "Failed" << " "
	   << std::right << std::setw(10) << "Error" << " "
	   << "Name" << "";

      for (unsigned int i = 0; i < pi->size(); ++i) {
	LogVerbatim("FwkSummary") << "TrigReport "
	     << std::right << std::setw( 5) << (trig_name_set_.find(pi->name()) != trig_name_set_.end())
	     << std::right << std::setw( 5) << pi->bitPosition() << " "
	     << std::right << std::setw(10) << pi->timesVisited(i) << " "
	     << std::right << std::setw(10) << pi->timesPassed(i) << " "
	     << std::right << std::setw(10) << pi->timesFailed(i) << " "
	     << std::right << std::setw(10) << pi->timesExcept(i) << " "
	     << pi->getWorker(i)->description().moduleLabel_ << "";
      }
    }

    pi=end_paths_.begin();
    pe=end_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TrigReport " << "------ Modules in End-Path: " << pi->name() << " ------------";
      LogVerbatim("FwkSummary") << "TrigReport "
	   << std::right << std::setw(10) << "Trig Bit#" << " "
	   << std::right << std::setw(10) << "Visited" << " "
	   << std::right << std::setw(10) << "Passed" << " "
	   << std::right << std::setw(10) << "Failed" << " "
	   << std::right << std::setw(10) << "Error" << " "
	   << "Name" << "";

      for (unsigned int i = 0; i < pi->size(); ++i) {
	LogVerbatim("FwkSummary") << "TrigReport "
	     << std::right << std::setw( 5) << (trig_name_set_.find(pi->name()) != trig_name_set_.end())
	     << std::right << std::setw( 5) << pi->bitPosition() << " "
	     << std::right << std::setw(10) << pi->timesVisited(i) << " "
	     << std::right << std::setw(10) << pi->timesPassed(i) << " "
	     << std::right << std::setw(10) << pi->timesFailed(i) << " "
	     << std::right << std::setw(10) << pi->timesExcept(i) << " "
	     << pi->getWorker(i)->description().moduleLabel_ << "";
      }
    }

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TrigReport " << "---------- Module Summary ------------";
    LogVerbatim("FwkSummary") << "TrigReport "
	 << std::right << std::setw(10) << "Visited" << " "
	 << std::right << std::setw(10) << "Run" << " "
	 << std::right << std::setw(10) << "Passed" << " "
	 << std::right << std::setw(10) << "Failed" << " "
	 << std::right << std::setw(10) << "Error" << " "
	 << "Name" << "";
    ai=workersBegin();
    ae=workersEnd();
    for(; ai != ae; ++ai) {
      LogVerbatim("FwkSummary") << "TrigReport "
	   << std::right << std::setw(10) << (*ai)->timesVisited() << " "
	   << std::right << std::setw(10) << (*ai)->timesRun() << " "
	   << std::right << std::setw(10) << (*ai)->timesPassed() << " "
	   << std::right << std::setw(10) << (*ai)->timesFailed() << " "
	   << std::right << std::setw(10) << (*ai)->timesExcept() << " "
	   << (*ai)->description().moduleLabel_ << "";

    }
    LogVerbatim("FwkSummary") << "";

    // The timing report (CPU and Real Time):

    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Event  Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport"
	 << std::setprecision(6) << std::fixed
	 << " CPU/event = " << timeCpuReal().first/std::max(1,totalEvents())
	 << " Real/event = " << timeCpuReal().second/std::max(1,totalEvents())
	 << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Path   Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per path-run "
	 << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    pi=trig_paths_.begin();
    pe=trig_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::setprecision(6) << std::fixed
	   << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1,totalEvents()) << " "
	   << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1,totalEvents()) << " "
	   << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1,pi->timesRun()) << " "
	   << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1,pi->timesRun()) << " "
	   << pi->name() << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per path-run "
	 << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "-------End-Path   Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per endpath-run "
	 << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    pi=end_paths_.begin();
    pe=end_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::setprecision(6) << std::fixed
	   << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1,totalEvents()) << " "
	   << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1,totalEvents()) << " "
	   << std::right << std::setw(10) << pi->timeCpuReal().first/std::max(1,pi->timesRun()) << " "
	   << std::right << std::setw(10) << pi->timeCpuReal().second/std::max(1,pi->timesRun()) << " "
	   << pi->name() << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per endpath-run "
	 << "";

    pi=trig_paths_.begin();
    pe=trig_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "---------- Modules in Path: " << pi->name() << " ---[sec]----";
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::right << std::setw(22) << "per event "
	   << std::right << std::setw(22) << "per module-visit "
	   << "";
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::right << std::setw(10) << "CPU" << " "
	   << std::right << std::setw(10) << "Real" << " "
	   << std::right << std::setw(10) << "CPU" << " "
	   << std::right << std::setw(10) << "Real" << " "
	   << "Name" << "";
      for (unsigned int i = 0; i < pi->size(); ++i) {
	LogVerbatim("FwkSummary") << "TimeReport "
	     << std::setprecision(6) << std::fixed
	     << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1,totalEvents()) << " "
	     << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1,totalEvents()) << " "
	     << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1,pi->timesVisited(i)) << " "
	     << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1,pi->timesVisited(i)) << " "
	     << pi->getWorker(i)->description().moduleLabel_ << "";
      }
    }
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per module-visit "
	 << "";

    pi=end_paths_.begin();
    pe=end_paths_.end();
    for(; pi != pe; ++pi) {
      LogVerbatim("FwkSummary") << "";
      LogVerbatim("FwkSummary") << "TimeReport " << "------ Modules in End-Path: " << pi->name() << " ---[sec]----";
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::right << std::setw(22) << "per event "
	   << std::right << std::setw(22) << "per module-visit "
	   << "";
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::right << std::setw(10) << "CPU" << " "
	   << std::right << std::setw(10) << "Real" << " "
	   << std::right << std::setw(10) << "CPU" << " "
	   << std::right << std::setw(10) << "Real" << " "
	   << "Name" << "";
      for (unsigned int i = 0; i < pi->size(); ++i) {
	LogVerbatim("FwkSummary") << "TimeReport "
	     << std::setprecision(6) << std::fixed
	     << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1,totalEvents()) << " "
	     << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1,totalEvents()) << " "
	     << std::right << std::setw(10) << pi->timeCpuReal(i).first/std::max(1,pi->timesVisited(i)) << " "
	     << std::right << std::setw(10) << pi->timeCpuReal(i).second/std::max(1,pi->timesVisited(i)) << " "
	     << pi->getWorker(i)->description().moduleLabel_ << "";
      }
    }
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per module-visit "
	 << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "TimeReport " << "---------- Module Summary ---[sec]----";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per module-run "
	 << std::right << std::setw(22) << "per module-visit "
	 << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    ai=workersBegin();
    ae=workersEnd();
    for(; ai != ae; ++ai) {
      LogVerbatim("FwkSummary") << "TimeReport "
	   << std::setprecision(6) << std::fixed
	   << std::right << std::setw(10) << (*ai)->timeCpuReal().first/std::max(1,totalEvents()) << " "
	   << std::right << std::setw(10) << (*ai)->timeCpuReal().second/std::max(1,totalEvents()) << " "
	   << std::right << std::setw(10) << (*ai)->timeCpuReal().first/std::max(1,(*ai)->timesRun()) << " "
	   << std::right << std::setw(10) << (*ai)->timeCpuReal().second/std::max(1,(*ai)->timesRun()) << " "
	   << std::right << std::setw(10) << (*ai)->timeCpuReal().first/std::max(1,(*ai)->timesVisited()) << " "
	   << std::right << std::setw(10) << (*ai)->timeCpuReal().second/std::max(1,(*ai)->timesVisited()) << " "
	   << (*ai)->description().moduleLabel_ << "";
    }
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << std::right << std::setw(10) << "CPU" << " "
	 << std::right << std::setw(10) << "Real" << " "
	 << "Name" << "";
    LogVerbatim("FwkSummary") << "TimeReport "
	 << std::right << std::setw(22) << "per event "
	 << std::right << std::setw(22) << "per module-run "
	 << std::right << std::setw(22) << "per module-visit "
	 << "";

    LogVerbatim("FwkSummary") << "";
    LogVerbatim("FwkSummary") << "T---Report end!" << "";
    LogVerbatim("FwkSummary") << "";

  }

  void Schedule::beginJob(EventSetup const& es) {
    AllWorkers::iterator i(workersBegin()),e(workersEnd());
    for(; i != e; ++i) { (*i)->beginJob(es); }

  }


  std::vector<ModuleDescription const*>
  Schedule::getAllModuleDescriptions() const {
    AllWorkers::const_iterator i(workersBegin());
    AllWorkers::const_iterator e(workersEnd());

    std::vector<ModuleDescription const*> result;
    result.reserve(all_workers_.size());

    for (; i!=e; ++i) {
      ModuleDescription const* p = (*i)->descPtr();
      result.push_back(p);
    }
    return result;
  }

  void
  Schedule::enableEndPaths(bool active) {
    endpathsAreActive_ = active;
  }

  bool
  Schedule::endPathsEnabled() const {
    return endpathsAreActive_;
  }

  void
  fillModuleInPathSummary(Path const&, 
			  ModuleInPathSummary&) {
  }


  void
  fillModuleInPathSummary(Path const& path, 
			  size_t which, 
			  ModuleInPathSummary& sum) {
    sum.timesVisited = path.timesVisited(which);
    sum.timesPassed  = path.timesPassed(which);
    sum.timesFailed  = path.timesFailed(which);
    sum.timesExcept  = path.timesExcept(which);
    sum.moduleLabel  = 
      path.getWorker(which)->description().moduleLabel_;
  }

  void 
  fillPathSummary(Path const& path, PathSummary& sum) {
    sum.name        = path.name();
    sum.bitPosition = path.bitPosition();
    sum.timesRun    = path.timesRun();
    sum.timesPassed = path.timesPassed();
    sum.timesFailed = path.timesFailed();
    sum.timesExcept = path.timesExcept();

    Path::size_type sz = path.size();
    std::vector<ModuleInPathSummary> temp(sz);
    for (size_t i = 0; i != sz; ++i) {
	fillModuleInPathSummary(path, i, temp[i]);
    }
    sum.moduleInPathSummaries.swap(temp);
  }

  void 
  fillWorkerSummaryAux(Worker const& w, WorkerSummary& sum) {
    sum.timesVisited = w.timesVisited();
    sum.timesRun     = w.timesRun();
    sum.timesPassed  = w.timesPassed();
    sum.timesFailed  = w.timesFailed();
    sum.timesExcept  = w.timesExcept();
    sum.moduleLabel  = w.description().moduleLabel_;
  }

  void
  fillWorkerSummary(Worker const* pw, WorkerSummary& sum) {
    fillWorkerSummaryAux(*pw, sum);
  }
  
  void
  Schedule::getTriggerReport(TriggerReport& rep) const {
    rep.eventSummary.totalEvents = totalEvents();
    rep.eventSummary.totalEventsPassed = totalEventsPassed();
    rep.eventSummary.totalEventsFailed = totalEventsFailed();

    fill_summary(trig_paths_,  rep.trigPathSummaries, &fillPathSummary);
    fill_summary(end_paths_,   rep.endPathSummaries,  &fillPathSummary);
    fill_summary(all_workers_, rep.workerSummaries,   &fillWorkerSummary);
  }

  void
  Schedule::resetAll() {
    for_each(workersBegin(), workersEnd(), boost::bind(&Worker::reset, _1));
    results_->reset();
    endpath_results_->reset();
  }

}
}

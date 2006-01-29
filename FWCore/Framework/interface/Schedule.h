#ifndef Framework_Schedule_H
#define Framework_Schedule_H 1

/*
	Author: Jim Kowalkowski  28-01-06

	$Id$

	A class for creating a schedule based on paths in the configuration file.
	The schedule is maintained as a sequence of paths.
	After construction, events can be fed to the object and passed through
	all the modules in the schedule.  All accounting about processing
	of events by modules and paths is contained here or in object held
	by containment.

	The trigger results producer and product are generated and managed here.
	This class also manages endpaths and calls to endjob and beginjob.
	Endpaths are just treated as a simple list of modules that need to
	do processing of the event and do not participate in trigger path
	activities.

	This class requires the high-level process pset.  It expects a pset
	with name @trigger_paths to be present and also uses @process_name.
	If the high-level pset contains an "options" pset, then the
	following optional parameters can be present:
	  bool wantSummary = true/false   # default false
	  bool makeTriggerResults = true/false # default false

	wantSummary indicates whether or not the pass/fail/error stats
	for modules and paths should be printed at the end-of-job or not.

	If makeTriggerResults is true, then a TriggerResults object will
	always be inserted into the event for any schedule.  If this is false,
	then a TriggerResults object will be placed into the event only if
	there is an instance of a filter module present in any path.  If
	TriggerResults are needed, the producer is always the first module
	in the endpath.  The TriggerResultInserter is given a fixed label
	of "TriggerResults".

	The Schedule prints a warning if output modules are present in paths.
	They belong in endpaths.  The Schedule moves them to the endpath.

	Processing of an event happens by pushing the event through the Paths.
	The scheduler performs the reset() on each of the workers independent
	of the Path objects. 

*/

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Framework/interface/TriggerResults.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/src/WorkerRegistry.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>
#include <set>

namespace edm
{
  class Schedule
  {
  public:
    Schedule(ParameterSet const& processDesc,
	     WorkerRegistry& wregistry,
	     ProductRegistry& pregistry,
	     ActionTable& actions,
	     boost::shared_ptr<ActivityRegistry> areg);
    ~Schedule();

    typedef std::vector<Path> TrigPaths;
    typedef std::set<Worker*> AllWorkers;
    typedef std::vector<Worker*> NonTrigPaths;
    typedef std::vector<Worker*> Workers;
    enum State { Ready=0, Running, Latched };

    void runOneEvent(EventPrincipal& eventPrincipal, EventSetup const& eventSetup);

    void beginJob(EventSetup const&);
    void endJob();

  private:
    void resetWorkers();
    void fillWorkers(const std::string& name, Workers& out);
    bool fillTrigPath(int bitpos,const std::string& name);
    void fillEndPath(const std::string& name);

    ParameterSet pset_;
    WorkerRegistry* worker_reg_;
    ProductRegistry* prod_reg_;
    ActionTable* act_table_;
    std::string proc_name_;
    ParameterSet trig_pset_;
    boost::shared_ptr<ActivityRegistry> act_reg_;

    State state_;
    std::vector<std::string> path_name_list_;
    std::vector<std::string> end_path_name_list_;
    boost::shared_ptr<TriggerResults::BitMask> results_;
    boost::shared_ptr<Worker> results_inserter_;
    AllWorkers all_workers_;
    TrigPaths trig_paths_;
    NonTrigPaths end_paths_;

    bool wantSummary_;
    bool makeTriggerResults_;
    int total_events_;
    int total_passed_;
  };
}

#endif

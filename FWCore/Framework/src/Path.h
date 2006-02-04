#ifndef Framework_Path_h
#define Framework_Path_h

/*

  Author: Jim Kowalkowski 28-01-06

  $Id: Path.h,v 1.1 2006/01/29 23:33:58 jbk Exp $

  An object of this type represents one path in a job configuration.
  It holds the assigned bit position and the list of workers that are
  an event must pass through when this parh is processed.  The workers
  are held in WorkerInPath wrappers so that per path execution statistics
  can be kept for each worker.

*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/src/WorkerInPath.h"
#include "FWCore/Framework/src/Worker.h"
#include "FWCore/Framework/interface/TriggerResults.h"

#include "boost/shared_ptr.hpp"
#include "boost/signal.hpp"

#include <string>
#include <vector>

namespace edm
{
  class EventPrincipal;
  class EventSetup;
  class ActionTable;
  class ActivityRegistry;

  class Path
  {
  public:
    typedef std::vector<WorkerInPath> Workers;
    typedef std::vector<Worker*> WorkerPtrs;
    typedef boost::shared_ptr<TriggerResults::BitMask> BitMaskPtr;
    typedef boost::shared_ptr<ActivityRegistry> ActivityRegistryPtr;

    Path(int bitpos, const std::string& path_name,
	 const Workers& workers,
	 BitMaskPtr bitmask,
	 ParameterSet const& proc_pset,
	 ActionTable& actions,
	 ActivityRegistryPtr reg);

    void runOneEvent(EventPrincipal&, EventSetup const&);

    bool passed() const { return pass_; }
    bool enabled() const { return enabled_; }
    bool invertDecision() const { return invertDecision_; }

    int bitPosition() const { return bitpos_; }
    std::string name() const { return name_; }

    int timesPassed() const { return timesPassed_; }
    int timesVisited() const { return timesVisited_; }

  private:
    int timesPassed_;
    int timesVisited_;
    bool enabled_;
    bool invertDecision_;
    bool pass_;
    int bitpos_;
    std::string name_;
    BitMaskPtr bitmask_;
    ActivityRegistryPtr act_reg_;
    ActionTable* act_table_;

    Workers workers_;
  };
}

#endif

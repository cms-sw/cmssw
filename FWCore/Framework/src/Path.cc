
#include "FWCore/Framework/src/Path.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/Log.h"

#include <string>

using namespace std;

namespace edm
{
  namespace
  {
    class CallPrePost
    {
    public:
      CallPrePost(ActivityRegistry* areg,const string& name, int bit):
	a_(areg),name_(name),bit_(bit)
      {
	//a_->prePathSignal(name_,bit_);
      }
      ~CallPrePost()
      {
	//a_->postPathSignal(name_,bit_);
      }
    private:
      ActivityRegistry* a_;
      std::string name_;
      int bit_;
    };
  }
  

  Path::Path(int bitpos, const std::string& path_name,
	     const Workers& workers,
	     BitMaskPtr bitmask,
	     ParameterSet const& proc_pset,
	     ActionTable& actions,
	     ActivityRegistryPtr areg):
    timesPassed_(),
    timesVisited_(),
    enabled_(true),
    invertDecision_(false),
    pass_(false),
    bitpos_(bitpos),
    name_(path_name),
    bitmask_(bitmask),
    act_reg_(areg),
    act_table_(&actions),
    workers_(workers)
  {
  }

#if 0
  void Worker::connect(ActivityRegistry::PrePath& pre,
		       ActivityRegistry::PostPath& post)
  {
    sigs_.prePathSignal.connect(pre);
    sigs_.postPathSignal.connect(post);
  }
#endif
  
  void Path::runOneEvent(EventPrincipal& ep, EventSetup const& es)
  {
    CallPrePost cpp(act_reg_.get(),name_,bitpos_);
    ++timesVisited_;
    Workers::iterator i(workers_.begin()),e(workers_.end());
    bool rc = true;
    for(;i!=e && rc==true;++i)
      {
	try
	  {
	    rc = i->runWorker(ep,es);
	  }
	catch(cms::Exception& e)
	  {
	    // there is no support as of yet for specific paths having
	    // different exception behavior

	    actions::ActionCodes code = act_table_->find(e.rootCause());

	    switch(code)
	      {
	      case actions::IgnoreCompletely:
		{
		  rc=true;
		  LogWarning(e.category())
		    << "Ignoring Exception in path " << name_
		    << ", message:\n" << e.what() << "\n";
		  break;
		}
	      case actions::FailPath:
		{
		  rc=false;
		  LogWarning(e.category())
		    << "Failing path " << name_
		    << ", due to exception, message:\n"
		    << e.what() << "\n";
		  break;
		}
	      default:
		{
		  LogError(e.category())
		    << "Exception going through path " << name_ << "\n";
		  throw edm::Exception(errors::ScheduleExecutionFailure,
				       "ProcessingStopped", e)
		    << "Exception going through path " << name_ << "\n";
		}
	      }
	  }
	catch(...)
	  {
	    LogError("PassingThrough")
	      << "Exception passing through path " << name_ << "\n";
	    throw;
	  }
      }

    if(rc) ++timesPassed_;
    (*bitmask_)[bitpos_]=rc;
  }

}

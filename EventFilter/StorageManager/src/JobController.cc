
#include "EventFilter/StorageManager/interface/JobController.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "IOPool/StreamerData/interface/Messages.h"
#include "IOPool/StreamerData/interface/StreamedProducts.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include "boost/bind.hpp"
#include "boost/thread/thread.hpp"
#include "boost/shared_ptr.hpp"

using namespace std;
using namespace edm;

using boost::thread;
using boost::bind;

/*
  the state are not being set properly
 */

namespace stor
{
  namespace 
  {
    string changeToPhony(const string& config)
    {
      return config;
    }
  }

  JobController::~JobController()
  {
  }

  JobController::JobController(const std::string& fu_config,
			       const std::string& my_config,
			       FragmentCollector::Deleter deleter)
  {
    // read trigger config
    // change to phony input source
    string new_config = changeToPhony(fu_config);
    setRegistry(new_config);
    init(my_config,deleter);
  } 

  JobController::JobController(const edm::ProductRegistry& reg,
			       const std::string& my_config,
			       FragmentCollector::Deleter d):
    prods_(reg)
  {
    init(my_config,d);
  }


  void JobController::init(const std::string& my_config,
			   FragmentCollector::Deleter deleter)
  {
    // JBK - 2/26/06 reordered the EPRunner and FragmentCollector
    // so that the fragment collector gets the registry from the 
    // EventProcessor instead of from the filter unit config

    std::auto_ptr<HLTInfo> inf(new HLTInfo(prods_));

    // ep takes ownership of inf!
    std::auto_ptr<EPRunner> ep(new EPRunner(my_config,inf));
    std::auto_ptr<FragmentCollector> 
      coll(new FragmentCollector(*(ep->getInfo()),deleter,
				 ep->getRegistry()
				 ));

    collector_.reset(coll.release());
    ep_runner_.reset(ep.release());
  }

  void JobController::setRegistry(const std::string& fu_config)
  {
    // generate a temporary event processor just to get out the
    // product registry
    EventProcessor tmp_ep(fu_config);
    InputSource& is = tmp_ep.getInputSource();
    prods_ = is.productRegistry();
  }

  void JobController::run(JobController* t)
  {
    t->processCommands();
  }

  void JobController::start()
  {
    // called from a differnt thread to start things going

    me_.reset(new boost::thread(boost::bind(JobController::run,this)));
    collector_->start();
    ep_runner_->start();
  }

  void JobController::stop()
  {
    // called from a different thread - trigger completion to the
    // job controller, which will cause a completion of the 
    // fragment collector and event processor

    edm::EventBuffer::ProducerBuffer cb(ep_runner_->getInfo()->getCommandQueue());
    MsgCode mc(cb.buffer(),MsgCode::DONE);
    mc.setCode(MsgCode::DONE);
    cb.commit(mc.codeSize());

    // should we wait here until the event processor and fragment
    // collectors are done?  Right now the wait is in the join.
  }

  void JobController::join()
  {
    // invoked from a different thread - block until "me_" is done
    me_->join();
  }

  void JobController::processCommands()
  {
    // called with this jobcontrollers own thread.
    // just wait for command messages now
    while(1)
      {
	edm::EventBuffer::ConsumerBuffer cb(ep_runner_->getInfo()->getCommandQueue());
	MsgCode mc(cb.buffer(),cb.size());

	if(mc.getCode()==MsgCode::DONE) break;

	// if this is an intialization message, then it is a new system
	// attempting to connect or an old system reconnecting
	// we must verify that the configuration in the HLTInfo object
	// is consistent with this new one.

	// right now we will ignore all messages
      }    

    // do not exit the thread until all subthreads are complete

    collector_->stop();
    collector_->join();
    ep_runner_->join();
  }
}

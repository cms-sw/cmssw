
#include "EventFilter/StorageManager/interface/JobController.h"

#include "boost/bind.hpp"

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

  JobController::JobController(const std::string& my_config,
			       log4cplus::Logger& applicationLogger,
			       FragmentCollector::Deleter deleter) :
  applicationLogger_(applicationLogger)      
  {
    // change to phony input source
    //string new_config = changeToPhony(fu_config);
    //setRegistry(new_config);
    init(my_config,deleter);
  } 

  void JobController::init(const std::string& my_config,
			   FragmentCollector::Deleter deleter)
  {
    std::auto_ptr<HLTInfo> inf(new HLTInfo());

    // ep takes ownership of inf!
    // what happens to ownership of inf with no EP_Runner??
    //std::auto_ptr<EPRunner> ep(new EPRunner(my_config,inf));
    //std::auto_ptr<FragmentCollector> 
    //  coll(new FragmentCollector(*(ep->getInfo()),deleter,
    //				 my_config));
    std::auto_ptr<FragmentCollector> 
	coll(new FragmentCollector(inf,deleter,applicationLogger_,
				   my_config));

    collector_.reset(coll.release());
    //ep_runner_.reset(ep.release());

    fileClosingTestInterval_ = 5;  // usually overwritten by SM
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
    //ep_runner_->start();
  }

  void JobController::stop()
  {
    // called from a different thread - trigger completion to the
    // job controller, which will cause a completion of the 
    // fragment collector and event processor

    //edm::EventBuffer::ProducerBuffer cb(ep_runner_->getInfo()->getCommandQueue());
    edm::EventBuffer::ProducerBuffer cb(collector_->getCommandQueue());
    MsgCode mc(cb.buffer(),MsgCode::DONE);
    mc.setCode(MsgCode::DONE);
    cb.commit(mc.codeSize());

    // should we wait here until the event processor and fragment
    // collectors are done?  Right now the wait is in the join.
  }

  void JobController::join()
  {
    // invoked from a different thread - block until "me_" is done
    if(me_) me_->join();
  }

  void JobController::processCommands()
  {
    time_t fileCheckIntervalStart = time(0);
    // called with this jobcontrollers own thread.
    // wait for command messages, and periodically send "file check"
    // messages to the FragmentCollector
    while(1)
      {
        // 02-Sep-2008, KAB: avoid the creation of a consumer buffer
        // (which blocks) if there are no messages on the queue
        if(!(collector_->getCommandQueue().empty()))
          {
            //edm::EventBuffer::ConsumerBuffer cb(ep_runner_->getInfo()->getCommandQueue());
            edm::EventBuffer::ConsumerBuffer cb(collector_->getCommandQueue());
            MsgCode mc(cb.buffer(),cb.size());

            if(mc.getCode()==MsgCode::DONE) break;

            // if this is an intialization message, then it is a new system
            // attempting to connect or an old system reconnecting
            // we must verify that the configuration in the HLTInfo object
            // is consistent with this new one.

            // right now we will ignore all messages
          }
        else
          {
            // sleep for a small amount of time, and then check if it is time
            // to check if files need closed.  If it is time for a check,
            // we send a special message to the FragmentCollector that tells
            // it to run the check.
            sleep(1);
            time_t now = time(0);
            if ((now - fileCheckIntervalStart) >= fileClosingTestInterval_) {
              fileCheckIntervalStart = now;
              EventBuffer::ProducerBuffer fragQBuff(getFragmentQueue());
              new (fragQBuff.buffer()) stor::FragEntry(0, 0, 0, 1, 1,
                                                       Header::FILE_CLOSE_REQUEST,
                                                       0, 0, 0, 0, 0);
              fragQBuff.commit(sizeof(stor::FragEntry));
            }
          }
      }    

    // do not exit the thread until all subthreads are complete

    collector_->stop();
    collector_->join();
    //ep_runner_->join();
  }
}

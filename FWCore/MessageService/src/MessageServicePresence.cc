// ----------------------------------------------------------------------
//
// MessageService Presence.cc
//
// Changes:
//
//   1 - 2/11/07  mf 	Added a call to edm::disableAllSigs(&oldset)
//			to disable signals so that they are all handled 
//			by the event processor thread
// 
//   2 - 8/10/09  mf 	Mods to support the use of abstract scribes
//		  cdj	so that standalones can work easily
//			
// 
//   2 - 8/12/09  mf 	Mods to get ownership of mlscribe better
//		  cdj	
//			
// 

#include "FWCore/MessageService/interface/MainThreadMLscribe.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageService/interface/MessageLoggerScribe.h"
#include "FWCore/MessageService/interface/ThreadQueue.h"

#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include <boost/bind.hpp>

using namespace edm::service;


namespace  {
void
  runMessageLoggerScribe(boost::shared_ptr<ThreadQueue> queue)
{
  sigset_t oldset;
  edm::disableAllSigs(&oldset);
  MessageLoggerScribe  m(queue);  
  m.run();
  // explicitly DO NOT reenableSigs(oldset) because -
  // 1) When this terminates, the main thread may not yet have done a join() and we
  //    don't want to handle the sigs at that point in this thread
  // 2) If we re-enable sigs, we will get the entire stack of accumulated ones (if any) 
}
}  // namespace

namespace edm {
namespace service {


MessageServicePresence::MessageServicePresence()
  : Presence()
  , m_queue (new ThreadQueue)
  , m_scribeThread
         ( ( (void) MessageLoggerQ::instance() // ensure Q's static data init'd
            , boost::bind(&runMessageLoggerScribe, m_queue)
	    			// start a new thread, run rMLS(m_queue)
				// ChangeLog 2
          ) ) 
	  // Note that m_scribeThread, which is a boost::thread, has a single-argument ctor - 
	  // just the function to be run.  But we need to do something first, namely,
	  // ensure that the MessageLoggerQ is in a valid state - and that requires
	  // a statement.  So we bundle that statement in parenthesis, separated by 
	  // a comma, with the argument we really want (runMessageLoggerScribe).  This
	  // creates a single argument, wheich evaluates to runMessageLoggerScribe after
	  // first executing the before-the-comma statement. 
{
  MessageLoggerQ::setMLscribe_ptr(
    boost::shared_ptr<edm::service::AbstractMLscribe>
        (new MainThreadMLscribe(m_queue))); 
    								// change log 3
  //std::cout << "MessageServicePresence ctor\n";
}


MessageServicePresence::~MessageServicePresence()
{
  MessageLoggerQ::MLqEND();
  m_scribeThread.join();
  MessageLoggerQ::setMLscribe_ptr
    (boost::shared_ptr<edm::service::AbstractMLscribe>());   // change log 3
}

} // end of namespace service  
} // end of namespace edm  

#ifndef HLT_FRAG_COLL_HPP
#define HLT_FRAG_COLL_HPP

/*
  All buffers passed in on queues are owned by the fragment collector.

  JBK - I think the frag_q is going to need to be a pair of pointers.
  The first is the address of the object that needs to be deleted 
  using the Deleter function.  The second is the address of the buffer
  of information that we manipulate (payload of the received object).

  The code should allow for deleters to be functors or simple functions.
 */

#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "IOPool/Streamer/interface/Utilities.h"
#include "FWCore/Framework/interface/ProductRegistry.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include <vector>
#include <map>

namespace stor
{

  class FragmentCollector
  {
  public:
    typedef void (*Deleter)(void*);
    typedef std::vector<unsigned char> Buffer;

    // This is not the most efficient way to store and manipulate this
    // type of data.  It is like this because there is not much time
    // available to create the prototype.
    typedef std::vector<FragEntry> Fragments;
    typedef std::map<edm::EventNumber_t, Fragments> Collection;

    FragmentCollector(const HLTInfo& h, Deleter d,
		      const edm::ProductRegistry& p);
    ~FragmentCollector();

    void start();
    void join();
    void stop();

    edm::EventBuffer& getFragmentQueue() { return *frag_q_; }
    
  private:
    static void run(FragmentCollector*);
    void processFragments();
    void processEvent(FragEntry* msg);

    edm::EventBuffer* cmd_q_;
    edm::EventBuffer* evtbuf_q_;
    edm::EventBuffer* frag_q_;

    Deleter buffer_deleter_;
    Buffer event_area_;
    Collection fragment_area_;
    edm::EventInserter inserter_;
    boost::shared_ptr<boost::thread> me_;
    const edm::ProductRegistry* prods_; // change to shared_ptr ? 
  };
}

#endif

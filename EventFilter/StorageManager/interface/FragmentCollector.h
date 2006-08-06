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
#include "DataFormats/Common/interface/ProductRegistry.h"
// added for Event Server by HWKC
#include "EventFilter/StorageManager/interface/EvtMsgRingBuffer.h"
// for hack
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerOutputService.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "boost/shared_ptr.hpp"
#include "boost/thread/thread.hpp"

#include <vector>
#include <map>
#include <string>
#include <fstream>

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
    void processHeader(FragEntry* msg);

    edm::EventBuffer* cmd_q_;
    edm::EventBuffer* evtbuf_q_;
    edm::EventBuffer* frag_q_;

    Deleter buffer_deleter_;
    Buffer event_area_;
    Collection fragment_area_;
    edm::EventInserter inserter_;
    boost::shared_ptr<boost::thread> me_;
    const edm::ProductRegistry* prods_; // change to shared_ptr ? 
	const stor::HLTInfo* info_;

    // hack here! HEREHEREHERE
    // should not need to set the bit counts as the first INIT messages has them
  public:
    void set_hlt_bit_count(uint32 count) { hlt_bit_cnt_ = count; }
    void set_l1_bit_count(uint32 count) { l1_bit_cnt_ = count; }
    void set_outoption(bool stream_only) { streamerOnly_ = stream_only; }
    void set_outfile(std::string outfilestart, unsigned long maxFileSize,
                     double highWaterMark, std::string path) 
                       { filen_ = outfilestart; 
                         maxFileSize_ = maxFileSize;
                         highWaterMark_ = highWaterMark;
                         path_ = path; }
  private:
    uint32 hlt_bit_cnt_;
    uint32 l1_bit_cnt_;
    bool streamerOnly_;
    std::string filen_;
    unsigned long maxFileSize_;
    double highWaterMark_;
    std::string path_;
    //ofstream ost_;
    std::auto_ptr<edm::StreamerOutputService> writer_;


  // added for Event Server by HWKC so SM can get events from ring buffer
  public:
    bool esbuf_isEmpty() { return evtsrv_area_.isEmpty(); }
    bool esbuf_isFull() { return evtsrv_area_.isFull(); }
//HEREHERE
    EventMsgView esbuf_pop_front() {return evtsrv_area_.pop_front();}
    void esbuf_push_back(EventMsgView msg) { evtsrv_area_.push_back(msg); }
//HEREHERE

    void set_esbuf_oneinN(int N) { oneinN_ = N; }

  private:
//HEREHERE
    stor::EvtMsgRingBuffer evtsrv_area_;
// the writer here
//  writer writer_;
//HEREHERE
    int oneinN_;  // place one in every oneinN_ events into the buffer
    int count_4_oneinN_;
  };
}

#endif

#ifndef ES_RING_BUFFER_H
#define ES_RING_BUFFER_H
// should really write as a template...
/*
  A ring buffer for the Storage Manager Event Server
  Serialized events are stored (EventMsg). The size of the
  buffer can be given in the constructor but the max size
  of each event is hardwired to 7MB (actually 7000000).
    This is used by the Storage Manager FragmentCollector.
  When the ring buffer is full it the oldest event gets
  overwritten.

  30 Mar 2006 - HWKC - First implementation.
*/

#include "IOPool/StreamerData/interface/Messages.h"

#include <vector>

// fix max size of event in each ring buffer element
#define MAX_EVTBUF_SIZE 7000000

namespace stor
{
  class ESRingBuffer
  {
    public:
    typedef std::vector<unsigned char> Buffer;

    ESRingBuffer(unsigned int size);
    virtual ~ESRingBuffer(){};

    edm::EventMsg pop_front();
    void push_back(edm::EventMsg msg);
    bool isEmpty();
    bool isFull();

    private:
    unsigned int maxsize_;  // max number of event slots in buffer
    int head_;       // next event to come off
    int tail_;       // last event put in
    int nextfree_;   // next free slot to fill in
    std::vector<Buffer> ring_buffer_; 
    std::vector<int> ring_totmsgsize_;
  };
} // end namespace stor

#endif



#include "IOPool/Streamer/interface/EventBuffer.h"
#include "boost/thread/thread.hpp"

#include <iostream>
#include <cstdlib>

using namespace edm;

// ---------------------------------------

struct Consumer
{
  Consumer(EventBuffer& b):b_(&b) { }

  void operator()();

  EventBuffer* b_;
};

void Consumer::operator()()
{
  while(1)
    {
      EventBuffer::ConsumerBuffer ob(*b_);
      if(ob.size()==0) break;
      //(int*)ob.buffer();
      int* i = (int*)ob.buffer();
      std::cout << "C" << *i << std::endl;
    }
}

// -----------------------

struct Producer
{
  Producer(EventBuffer& b, int total):b_(&b),total_(total) { }

  void operator()();

  EventBuffer* b_;
  int total_;
};

void Producer::operator()()
{
  for(int i = 0; i < total_; ++i) {
      //boost::thread::yield();
      for(int j = 0; j<(1<<17); ++j);
      EventBuffer::ProducerBuffer ib(*b_);
      int* v = (int*)ib.buffer();
      *v = i;
      //cout << "P" << i << std::endl;
      ib.commit(sizeof(int));
  }

  EventBuffer::ProducerBuffer ib(*b_);
  ib.commit(0);

}

int main(int argc, char* argv[])
{
  if(argc<3)
    {
      std::cerr << "usage: " << argv[0] << " event_size queue_depth number_to_gen"
	   << std::endl;
      return -1;
    }

  int event_sz = atoi(argv[1]);
  int queue_dep = atoi(argv[2]);
  int total = atoi(argv[3]);

  EventBuffer buf(event_sz,queue_dep);
  Consumer c(buf);
  Producer p(buf,total);
  boost::thread con(c);
  boost::thread pro(p);

  pro.join();
  con.join();
  return 0;
}

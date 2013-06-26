
#include "IOPool/Streamer/interface/EventBuffer.h"
#include "boost/thread/thread.hpp"
#include "boost/bind.hpp"
#include "boost/signal.hpp"

#include <iostream>
#include <cstdlib>

using namespace edm;

// ---------------------------------------

typedef boost::signal<void (int)> DataSignal;
typedef boost::signals::connection Connection;

struct Consumer
{
  Consumer(EventBuffer& b, DataSignal& c);
  ~Consumer() { c_.disconnect(); }

  void operator()();

  void callme(int value);

  EventBuffer* b_;
  Connection c_;
};

Consumer::Consumer(EventBuffer& b, DataSignal& s):
  b_(&b),
  c_(s.connect(boost::bind(&Consumer::callme,this,_1)))
{

}

void Consumer::operator()()
{
  while(1)
    {
      EventBuffer::ConsumerBuffer ob(*b_);
      if(ob.size()==0) break;
      int* i = (int*)ob.buffer();
      std::cout << "C" << *i << std::endl;
    }
}

void Consumer::callme(int value)
{
  EventBuffer::ProducerBuffer ib(*b_);
  int* v = (int*)ib.buffer();
  *v = value;
  std::cout << "P" << value << std::endl;

  if(value==-1)
    ib.commit(0);
  else
    ib.commit(sizeof(int));  
}

// -----------------------

struct Producer
{
  Producer(EventBuffer& b, int total):b_(&b),total_(total) { }

  void operator()();

  EventBuffer* b_;
  int total_;

  DataSignal sig_;
};


void Producer::operator()()
{
  for(int i = 0; i < total_; ++i)
    {
      //boost::thread::yield();
      for(int j = 0; j<(1<<17); ++j);
      /*
      EventBuffer::ProducerBuffer ib(*b_);
      int* v = (int*)ib.buffer();
      *v = i;
      std::cout << "P" << i << std::endl;
      ib.commit(sizeof(int));
      */
      sig_(i);
    }

  sig_(-1);
  sig_(-1);
  //EventBuffer::ProducerBuffer ib(*b_);
  //ib.commit(0);

}

void prunner(Producer* p)
{
  (*p)();
  std::cout << "PD" << std::endl;
}

void crunner(Consumer* c)
{
  (*c)();
  std::cout << "CD" << std::endl;
}

int main(int argc, char* argv[])
{
  if(argc<3)
    {
      std::cerr << "usage: " << argv[0] << " event_size queue_depth number_to_gen"
	   << std::endl;
      return -1;
    }

  std::cout << "continuing" << std::endl;

  int event_sz = atoi(argv[1]);
  int queue_dep = atoi(argv[2]);
  int total = atoi(argv[3]);

  std::cout << "(1)" << std::endl;
  EventBuffer buf(event_sz,queue_dep);
  Producer p(buf,total);
  Consumer c(buf,p.sig_),c2(buf,p.sig);
  std::cout << "(2)" << std::endl;
  boost::thread con(boost::bind(crunner,&c));
  boost::thread con2(boost::bind(crunner,&c2));
  boost::thread pro(boost::bind(prunner,&p));
  std::cout << "(3)" << std::endl;

  con.join();
  con2.join();
  pro.join();
  return 0;
}

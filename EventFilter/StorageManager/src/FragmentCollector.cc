
#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/test/SillyLockService.h"
#include "IOPool/StreamerData/interface/Messages.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "boost/bind.hpp"

#include <algorithm>
#include <utility>
#include <cstdlib>

using namespace edm;
using namespace std;

static const bool debugme = getenv("FRAG_DEBUG")!=0;  
#define FR_DEBUG if(debugme) std::cerr

namespace stor
{

  FragmentCollector::FragmentCollector(const HLTInfo& h,Deleter d,
				       const ProductRegistry& p):
    cmd_q_(&(h.getCommandQueue())),
    evtbuf_q_(&(h.getEventQueue())),
    frag_q_(&(h.getFragmentQueue())),
    buffer_deleter_(d),
    event_area_(1000*1000*7),
    inserter_(*evtbuf_q_),
    prods_(&p),
	info_(&h)
  {
  }

  FragmentCollector::~FragmentCollector()
  {
  }

  void FragmentCollector::run(FragmentCollector* t)
  {
    t->processFragments();
  }

  void FragmentCollector::start()
  {
    me_.reset(new boost::thread(boost::bind(FragmentCollector::run,this)));
  }

  void FragmentCollector::join()
  {
    me_->join();
  }

  void FragmentCollector::processFragments()
  {
    // everything comes in on the fragment queue, even
    // command-like messages.  we need to dispatch things
    // we recogize - either execute the command, forward it
    // to the command queue, or process it for output to the 
    // event queue.
    bool done=false;

    while(!done)
      {
	EventBuffer::ConsumerBuffer cb(*frag_q_);
	if(cb.size()==0) break;
	FragEntry* entry = (FragEntry*)cb.buffer();
	FR_DEBUG << "FragColl: " << (void*)this << " Got a buffer size="
		 << entry->buffer_size_ << endl;
	MsgCode mc(entry->buffer_address_,entry->buffer_size_);
	
	switch(mc.getCode())
	  {
	  case MsgCode::EVENT:
	    {
	      FR_DEBUG << "FragColl: Got an Event" << endl;
	      processEvent(entry);
	      break;
	    }
	  case MsgCode::DONE:
	    {
	      // make sure that this is actually sent by the controller! (JBK)
	      FR_DEBUG << "FragColl: Got a Done" << endl;
	      done=true;
	      break;
	    }
	  default:
	    {
	      FR_DEBUG << "FragColl: Got junk" << endl;
	      break; // lets ignore other things for now
	    }
	  }
      }
    
	FR_DEBUG << "FragColl: DONE!" << endl;
    edm::EventBuffer::ProducerBuffer cb(*evtbuf_q_);
	long* vp = (long*)cb.buffer();
	*vp=0;
	cb.commit(sizeof(long));
  }

  void FragmentCollector::stop()
  {
    // called from a different thread - trigger completion to the
    // fragment collector, which will cause a completion of the 
    // event processor

    edm::EventBuffer::ProducerBuffer cb(*frag_q_);
    MsgCode mc(cb.buffer(),MsgCode::DONE);
    mc.setCode(MsgCode::DONE);
    // cb.commit(mc.totalSize());
    cb.commit();
  }

  void FragmentCollector::processEvent(FragEntry* entry)
  {
    EventMsg msg(entry->buffer_address_,entry->buffer_size_);

    if(msg.getTotalSegs()==1)
      {
	FR_DEBUG << "FragColl: Got an Event with one segment" << endl;
	FR_DEBUG << "FragColl: Event size " << entry->buffer_size_ << endl;
	FR_DEBUG << "FragColl: Event ID " << msg.getEventNumber() << endl;
	// send immediately
	boost::mutex::scoped_lock sl(info_->getExtraLock());
	inserter_.insert(msg,*prods_);
	// is the buffer properly released (deleted)? (JBK)
	(*buffer_deleter_)(entry);
	return;
      }

    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(msg.getEventNumber(),Fragments()));
    
    rc.first->second.push_back(*entry);
    FR_DEBUG << "FragColl: added fragment" << endl;
    
    if((int)rc.first->second.size()==msg.getTotalSegs())
      {
	FR_DEBUG << "FragColl: completed an event with "
		 << msg.getTotalSegs() << " segments" << endl;
	// we are done with this event
	// assemble parts
	EventMsg em(&event_area_[0],event_area_.size(),
		    msg.getEventNumber(),msg.getRunNumber(),
		    1,1);
	unsigned char* pos = (unsigned char*)em.data();
	
	int sum=0;
	Fragments::iterator
	  i(rc.first->second.begin()),e(rc.first->second.end());

	for(;i!=e;++i)
	  {
	    EventMsg frag(i->buffer_address_,i->buffer_size_);
	    int dsize = frag.getDataSize();
	    sum+=dsize;
	    unsigned char* from=(unsigned char*)frag.data();
	    copy(from,from+dsize,pos);
	    // ask deleter to kill off the buffer
	    (*buffer_deleter_)(i->buffer_object_);
	  }
	em.setDataSize(sum);
	boost::mutex::scoped_lock sl(info_->getExtraLock());
	inserter_.insert(em,*prods_);
	// remove the entry from the map
	fragment_area_.erase(rc.first);
      }
  }
}

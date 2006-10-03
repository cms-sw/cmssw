
#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/test/SillyLockService.h"
#include "IOPool/Streamer/interface/Messages.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/StreamTranslator.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"

//#include "IOPool/Streamer/interface/DumpTools.h"

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
				       const ProductRegistry& p,
                                       const string& config_str):
    cmd_q_(&(h.getCommandQueue())),
    evtbuf_q_(&(h.getEventQueue())),
    frag_q_(&(h.getFragmentQueue())),
    buffer_deleter_(d),
    event_area_(1000*1000*7),
    inserter_(*evtbuf_q_),
    prods_(&p),
	info_(&h), 
    runNumber_(0),maxFileSize_(1073741824), highWaterMark_(0.9),
    writer_(new edm::StreamerOutSrvcManager(config_str)),
    evtsrv_area_(10),
    oneinN_(10), count_4_oneinN_(0) // added for Event Server by HWKC
  {
    // supposed to have given parameterSet smConfigString to writer_
    // at ctor
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
//HEREHERE
	//MsgCode mc(entry->buffer_address_,entry->buffer_size_);
        // cannot use this as we don't have a header in each fragment
	//HeaderView mc(entry->buffer_address_);
	
//HEREHERE
	//switch(mc.getCode())
	//switch(mc.code())
	switch(entry->code_)
	  {
//HEREHERE
	  case Header::EVENT:
	    {
	      FR_DEBUG << "FragColl: Got an Event" << endl;
	      processEvent(entry);
	      break;
	    }
//HEREHERE
	  case Header::DONE:
	    {
	      // make sure that this is actually sent by the controller! (JBK)
	      FR_DEBUG << "FragColl: Got a Done" << endl;
	      done=true;
	      break;
	    }
// HEREHERE put in test for INIT message and write out INIT message
//
	  case Header::INIT:
	    {
	      FR_DEBUG << "FragColl: Got an Init" << endl;
	      processHeader(entry);
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
//HEREHERE  - ending loop
	cb.commit(sizeof(long));
//HEREHEREHERE
        // is this the right place to close the file??
         //FR_DEBUG << "FragColl: Received halt message. Closing file " << std::endl;
   // need an EOF record?
   /*
   unsigned long dummyrun = 0;
   unsigned long dummystored = 1000;
     uint32 dummyStatusCode = 1234;
    std::vector<uint32> hltStats;

    hltStats.push_back(32);
    hltStats.push_back(33);
    hltStats.push_back(34);
    uint64 dummy_first_event_offset_ = 0;
    uint64 dummy_last_event_offset_ = 10000;
     //HEREHERE need to change this
    EOFRecordBuilder eof(dummyrun,
                         dummystored,
                         dummyStatusCode,
                         hltStats,
                         dummy_first_event_offset_,
                         dummy_last_event_offset_);
  ost_.write((const char*)eof.recAddress(), eof.size() );

       ost_.close();
    */
    // note that file is not closed until the writers inside
    // writer_ is destroyed
    if(streamerOnly_)  writer_->stop();
    // WE DO NOT want a crash if halt straight after ready HEREHEREHERE
    //if(streamerOnly_ && writer_->get_currfile()!="") writer_->stop();
    
  }

  void FragmentCollector::stop()
  {
    // called from a different thread - trigger completion to the
    // fragment collector, which will cause a completion of the 
    // event processor

    edm::EventBuffer::ProducerBuffer cb(*frag_q_);
    // MsgCode mc(cb.buffer(),MsgCode::DONE);
    // mc.setCode(MsgCode::DONE);
    // cb.commit(mc.totalSize());
//HEREHERE
    cb.commit();
  }

  void FragmentCollector::processEvent(FragEntry* entry)
  {
//HEREHERE
    //EventMsg msg(entry->buffer_address_,entry->buffer_size_);

//HEREHERE
    //if(msg.getTotalSegs()==1)
    if(entry->totalSegs_==1)
      {
	FR_DEBUG << "FragColl: Got an Event with one segment" << endl;
	FR_DEBUG << "FragColl: Event size " << entry->buffer_size_ << endl;
//HEREHERE
	//FR_DEBUG << "FragColl: Event ID " << msg.getEventNumber() << endl;
	FR_DEBUG << "FragColl: Event ID " << entry->id_ << endl;

//HEREHERE from here
	// send immediately
        EventMsgView emsg(entry->buffer_address_, hlt_bit_cnt_, l1_bit_cnt_ );
        // do the if here for streamer file writing
      if(!streamerOnly_)
      {
	std::auto_ptr<edm::EventPrincipal> evtp;
	{
	  boost::mutex::scoped_lock sl(info_->getExtraLock());
	  //evtp = inserter_.decode(msg,*prods_);
          evtp = StreamTranslator::deserializeEvent(emsg, *prods_);
	  //evtp = inserter_.decode(msg,*prods_);
	}
	inserter_.send(evtp);
      } else {
// put code to write to streamer file here
//    writer_.write(emsg);
          FR_DEBUG << "FragColl: writing event size " << entry->buffer_size_ << endl;
          //ost_.write((const char*)entry->buffer_address_, entry->buffer_size_);
          writer_->manageEventMsg(emsg);
      }

//HEREHERE to here
        // added for Event Server by HWKC - copy event to Event Server buffer
        count_4_oneinN_++;
        if(count_4_oneinN_ == oneinN_)
        {
//HEREHERE 
          evtsrv_area_.push_back(emsg);
          count_4_oneinN_ = 0;
        }
        if (eventServer_.get() != NULL)
        {
          eventServer_->processEvent(emsg);
        }

	// used to be like next line, but that does decode and send in
	// one operation, which can cause deadlock with the global junky lock
	// inserter_.insert(msg,*prods_);

	// is the buffer properly released (deleted)? (JBK)
	(*buffer_deleter_)(entry);
	return;
      }

//HEREHERE
    //pair<Collection::iterator,bool> rc =
    //  fragment_area_.insert(make_pair(msg.getEventNumber(),Fragments()));
    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(entry->id_,Fragments()));
    
    rc.first->second.push_back(*entry);
    FR_DEBUG << "FragColl: added fragment" << endl;
    
//HEREHERE
    //if((int)rc.first->second.size()==msg.getTotalSegs())
    if((int)rc.first->second.size()==entry->totalSegs_)
      {
//HEREHERE
	FR_DEBUG << "FragColl: completed an event with "
		 << entry->totalSegs_ << " segments" << endl;
	// we are done with this event
	// assemble parts
//HEREHERE from here
	//EventMsg em(&event_area_[0],event_area_.size(),
//		    msg.getEventNumber(),msg.getRunNumber(),
//		    1,1);
	//unsigned char* pos = (unsigned char*)em.data();
	unsigned char* pos = (unsigned char*)&event_area_[0];
//HEREHERE to here
	
	int sum=0;
	unsigned int lastpos=0;
	Fragments::iterator
	  i(rc.first->second.begin()),e(rc.first->second.end());

	for(;i!=e;++i)
	  {
//HEREHERE from here
	    //EventMsg frag(i->buffer_address_,i->buffer_size_);
	    //int dsize = frag.getDataSize();
	    int dsize = i->buffer_size_;
	    sum+=dsize;
	    //unsigned char* from=(unsigned char*)frag.data();
	    unsigned char* from=(unsigned char*)i->buffer_address_;
	    ////copy(from,from+dsize,pos);
	    //copy(i->buffer_address_,i->buffer_address_+i->buffer_size_,pos+lastpos);
	    copy(from,from+dsize,pos+lastpos);
	    //copy(i->buffer_address_,i->buffer_address_+dsize ,pos+lastpos);
	    //copy(from,from+dsize,pos+lastpos);
            //lastpos = lastpos + dsize;
            //lastpos = lastpos + i->buffer_size_;
            lastpos = lastpos + dsize;
//HEREHERE to here
	    // ask deleter to kill off the buffer
	    //(*buffer_deleter_)(i->buffer_object_);
	    (*buffer_deleter_)(&(*i));
	  }

//HEREHERE from here
	//em.setDataSize(sum);
        EventMsgView emsg(&event_area_[0], hlt_bit_cnt_, l1_bit_cnt_);
      if(!streamerOnly_)
      {
	std::auto_ptr<edm::EventPrincipal> evtp;
	{
	  boost::mutex::scoped_lock sl(info_->getExtraLock());
	  //evtp = inserter_.decode(em,*prods_);
          evtp = StreamTranslator::deserializeEvent(emsg, *prods_);
	}
	inserter_.send(evtp);
      } else {
// put code to write to streamer file here
//    writer_.write(emsg);
          FR_DEBUG << "FragColl: writing event size " << sum << endl;
          //ost_.write((const char*)&event_area_[0], sum);
          writer_->manageEventMsg(emsg);
      }

//HEREHERE to here
        // added for Event Server by HWKC - copy event to Event Server buffer
        // note that em does not have the correct totalsize in totalSize()
        // the ring buffer must use msgSize() or we send always 7MB events
        count_4_oneinN_++;
        if(count_4_oneinN_ == oneinN_)
        {
          evtsrv_area_.push_back(emsg);
          count_4_oneinN_ = 0;
        }
        if (eventServer_.get() != NULL)
        {
          eventServer_->processEvent(emsg);
        }

	// see inserter use comment above
	// inserter_.insert(em,*prods_);

	// remove the entry from the map
	fragment_area_.erase(rc.first);
      }
  }
  void FragmentCollector::processHeader(FragEntry* entry)
  {
//HEREHERE
    InitMsgView msg(entry->buffer_address_);

//HEREHERE
    //if(entry->totalSegs_==1) // should test if these are fragments
    // currently these are taken from the already combined registry
    // fragments if any - need to change where the fragments are
    // queued here and remade here
   
    // open file here as there is only one of these per run
    //std::string outfilename = filename_ + ".dat";
    FR_DEBUG << "FragmentCollector: streamer file starting with " << filen_ << endl;
    //ost_.open(outfilename.c_str(),ios_base::binary | ios_base::out);
    //if(!ost_)
    //{
    //// throw exceptions in the online?
    ////throw cms::Exception("Configuration","simpleI2OReceiver")
    ////  << "cannot open file " << filename_;
    //  std::cerr << "FragmentCollector: Cannot open file " << outfilename << std::endl;
    //}
    FR_DEBUG << "FragColl: writing INIT size " << entry->buffer_size_ << endl;
    //ost_.write((const char*)entry->buffer_address_, entry->buffer_size_);
    //dumpInitHeader(&msg);
    // should be passing smConfigSTring to writer_ at construction
    writer_->manageInitMsg(filen_, runNumber_, maxFileSize_, highWaterMark_, path_, mpath_, catalog_, disks_, msg);
  }
}

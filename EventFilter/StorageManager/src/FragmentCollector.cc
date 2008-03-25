// $Id: FragmentCollector.cc,v 1.33 2007/05/16 22:55:03 hcheung Exp $

#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/interface/ProgressMarker.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "IOPool/Streamer/interface/Messages.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/StreamDQMDeserializer.h"

#include "boost/bind.hpp"

#include <algorithm>
#include <utility>
#include <cstdlib>
#include <fstream>

using namespace edm;
using namespace std;

static const bool debugme = getenv("FRAG_DEBUG")!=0;  
#define FR_DEBUG if(debugme) std::cerr


namespace stor
{

// TODO fixme: this quick fix to give thread status should be reviewed!
  struct SMFC_thread_data
  {
    SMFC_thread_data() {
      exception_in_thread = false;
      reason_for_exception = "";
    }

    volatile bool exception_in_thread;
    std::string reason_for_exception;
  };

  static SMFC_thread_data SMFragCollThread;

  bool getSMFC_exceptionStatus() { return SMFragCollThread.exception_in_thread; }
  std::string getSMFC_reason4Exception() { return SMFragCollThread.reason_for_exception; }


  FragmentCollector::FragmentCollector(HLTInfo& h,Deleter d,
                                       const string& config_str):
    cmd_q_(&(h.getCommandQueue())),
    evtbuf_q_(&(h.getEventQueue())),
    frag_q_(&(h.getFragmentQueue())),
    buffer_deleter_(d),
    prods_(0),
    info_(&h), 
    writer_(new edm::ServiceManager(config_str)),
    dqmServiceManager_(new stor::DQMServiceManager())
  {
    // supposed to have given parameterSet smConfigString to writer_
    // at ctor
    event_area_.reserve(7000000);
  }
  FragmentCollector::FragmentCollector(std::auto_ptr<HLTInfo> info,Deleter d,
                                       const string& config_str):
    cmd_q_(&(info.get()->getCommandQueue())),
    evtbuf_q_(&(info.get()->getEventQueue())),
    frag_q_(&(info.get()->getFragmentQueue())),
    buffer_deleter_(d),
    prods_(0),
    info_(info.get()), 
    writer_(new edm::ServiceManager(config_str)),
    dqmServiceManager_(new stor::DQMServiceManager())
  {
    // supposed to have given parameterSet smConfigString to writer_
    // at ctor
    event_area_.reserve(7000000);
  }

  FragmentCollector::~FragmentCollector()
  {
  }

  void FragmentCollector::run(FragmentCollector* t)
  {
    try {
      t->processFragments();
    }
    catch(cms::Exception& e)
    {
       edm::LogError("StorageManager") << "Fatal error in FragmentCollector thread: " << "\n"
                 << e.explainSelf() << std::endl;
       SMFragCollThread.exception_in_thread = true;
       SMFragCollThread.reason_for_exception = "Fatal error in FragmentCollector thread: \n" +
          e.explainSelf();
    }
    // just let the thread end here there is no cleanup, no recovery possible
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
    // we recognize - either execute the command, forward it
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
	switch(entry->code_)
	  {
	  case Header::EVENT:
	    {
	      FR_DEBUG << "FragColl: Got an Event" << endl;
	      processEvent(entry);
	      break;
	    }
	  case Header::DONE:
	    {
	      // make sure that this is actually sent by the controller! (JBK)
              // this does nothing currently
	      FR_DEBUG << "FragColl: Got a Done" << endl;
	      done=true;
	      break;
	    }
	  case Header::INIT:
	    {
	      FR_DEBUG << "FragColl: Got an Init" << endl;
	      processHeader(entry);
	      break;
	    }
	  case Header::DQM_EVENT:
	    {
	      FR_DEBUG << "FragColl: Got a DQM_Event" << endl;
	      processDQMEvent(entry);
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
    writer_->stop();
    dqmServiceManager_->stop();
  }

  void FragmentCollector::stop()
  {
    // called from a different thread - trigger completion to the
    // fragment collector, which will cause a completion of the 
    // event processor

    edm::EventBuffer::ProducerBuffer cb(*frag_q_);
    cb.commit();
  }

  void FragmentCollector::processEvent(FragEntry* entry)
  {
    ProgressMarker::instance()->processing(true);
    if(entry->totalSegs_==1)
    {
	FR_DEBUG << "FragColl: Got an Event with one segment" << endl;
	FR_DEBUG << "FragColl: Event size " << entry->buffer_size_ << endl;
	FR_DEBUG << "FragColl: Event ID " << entry->id_ << endl;

	// send immediately
        EventMsgView emsg(entry->buffer_address_);
        FR_DEBUG << "FragColl: writing event size " << entry->buffer_size_ << endl;
        writer_->manageEventMsg(emsg);

        if (eventServer_.get() != NULL)
        {
          eventServer_->processEvent(emsg);
        }

	// make sure the buffer properly released
	(*buffer_deleter_)(entry);
	return;
    } // end of single segment test

    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(FragKey(entry->code_, entry->run_, entry->id_, 0), Fragments()));
    
    rc.first->second.push_back(*entry);
    FR_DEBUG << "FragColl: added fragment" << endl;
    
    if((int)rc.first->second.size()==entry->totalSegs_)
    {
	FR_DEBUG << "FragColl: completed an event with "
		 << entry->totalSegs_ << " segments" << endl;
        // we are done with this event so assemble parts
        // but first make sure we have enough room; use an overestimate
        unsigned int max_sizePerFrame = rc.first->second.begin()->buffer_size_;
        if((entry->totalSegs_ * max_sizePerFrame) > event_area_.capacity()) {
          event_area_.resize(entry->totalSegs_ * max_sizePerFrame);
        }
        unsigned char* pos = (unsigned char*)&event_area_[0];
	
	int sum=0;
	unsigned int lastpos=0;
	Fragments::iterator
	  i(rc.first->second.begin()),e(rc.first->second.end());

	for(;i!=e;++i)
	{
	    int dsize = i->buffer_size_;
	    sum+=dsize;
	    unsigned char* from=(unsigned char*)i->buffer_address_;
	    copy(from,from+dsize,pos+lastpos);
            lastpos = lastpos + dsize;
	    // ask deleter to kill off the buffer
	    (*buffer_deleter_)(&(*i));
	}

        EventMsgView emsg(&event_area_[0]);
        FR_DEBUG << "FragColl: writing event size " << sum << endl;
        writer_->manageEventMsg(emsg);

        if (eventServer_.get() != NULL)
        {
          eventServer_->processEvent(emsg);
        }

	// remove the entry from the map
	fragment_area_.erase(rc.first);
    }
    ProgressMarker::instance()->processing(false);
  }

  void FragmentCollector::processHeader(FragEntry* entry)
  {
    // This does not yet handle fragmented INIT messages, so one should
    // probably really test for entry->totalSegs_==1 in case fragments
    // are passed through. (Should eventually handle fragments as the
    // fragment queue design was to make it thread-safe. Currently
    // any fragmented INIT messages are done differently).

    InitMsgView msg(entry->buffer_address_);

    FR_DEBUG << "FragColl: writing INIT size " << entry->buffer_size_ << endl;

    writer_->manageInitMsg(catalog_, disks_, sourceId_, msg);
  }

  void FragmentCollector::processDQMEvent(FragEntry* entry)
  {
    ProgressMarker::instance()->processing(true);
    if(entry->totalSegs_==1)
    {
      FR_DEBUG << "FragColl: Got a DQM_Event with one segment" << endl;
      FR_DEBUG << "FragColl: DQM_Event size " << entry->buffer_size_ << endl;
      FR_DEBUG << "FragColl: DQM_Event ID " << entry->id_ << endl;
      FR_DEBUG << "FragColl: DQM_Event folderID " << entry->folderid_ << endl;

      DQMEventMsgView dqmEventView(entry->buffer_address_);
      /*  do not deserialize even for debug output but keep this for now
      // temporary debug output
      std::cout << "  DQM Message data:" << std::endl; 
      std::cout << "    protocol version = "
                << dqmEventView.protocolVersion() << std::endl; 
      std::cout << "    header size = "
                << dqmEventView.headerSize() << std::endl; 
      std::cout << "    run number = "
                << dqmEventView.runNumber() << std::endl; 
      std::cout << "    event number = "
                << dqmEventView.eventNumberAtUpdate() << std::endl; 
      std::cout << "    lumi section = "
                << dqmEventView.lumiSection() << std::endl; 
      std::cout << "    update number = "
                << dqmEventView.updateNumber() << std::endl; 
      std::cout << "    compression flag = "
                << dqmEventView.compressionFlag() << std::endl; 
      std::cout << "    reserved word = "
                << dqmEventView.reserved() << std::endl; 
      std::cout << "    release tag = "
                << dqmEventView.releaseTag() << std::endl; 
      std::cout << "    top folder name = "
                << dqmEventView.topFolderName() << std::endl; 
      std::cout << "    sub folder count = "
                << dqmEventView.subFolderCount() << std::endl; 
      edm::StreamDQMDeserializer deserializeWorker;
      std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
        deserializeWorker.deserializeDQMEvent(dqmEventView);
      DQMEvent::TObjectTable::const_iterator toIter;
      for (toIter = toTablePtr->begin();
           toIter != toTablePtr->end(); toIter++) {
        std::string subFolderName = toIter->first;
        std::cout << "  folder = " << subFolderName << std::endl;
        std::vector<TObject *> toList = toIter->second;
        for (int tdx = 0; tdx < (int) toList.size(); tdx++) {
          TObject *toPtr = toList[tdx];
          string cls = toPtr->IsA()->GetName();
          string nm = toPtr->GetName();
          std::cout << "    TObject class = " << cls
                    << ", name = " << nm << std::endl;
        }
      }
      */
      // Manage this DQMEvent, temporarily just stick it in the DQMEventServer
      //if (DQMeventServer_.get() != NULL)
      //{
      //  DQMeventServer_->processDQMEvent(dqmEventView);
      //}
      dqmServiceManager_->manageDQMEventMsg(dqmEventView);

      // do the appropriate thing with this DQM_Event
      //std::cout << "FragColl: Got a DQM_Event with one segment" 
      //          << " DQM_Event size " << entry->buffer_size_
      //          << " DQM_Event ID " << entry->id_
      //          << " DQM_Event folderID " << entry->folderid_ << std::endl;
      // properly release (delete) the buffer
      (*buffer_deleter_)(entry);
      return;
    } // end of single segment test

    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(FragKey(entry->code_, entry->run_, entry->id_, entry->folderid_), Fragments()));
    
    rc.first->second.push_back(*entry);
    FR_DEBUG << "FragColl: added DQM fragment" << endl;
    
    if((int)rc.first->second.size()==entry->totalSegs_)
    {
      FR_DEBUG << "FragColl: completed a DQM_event with "
       << entry->totalSegs_ << " segments" << endl;
      // we are done with this event so assemble parts
      // but first make sure we have enough room; use an overestimate
      unsigned int max_sizePerFrame = rc.first->second.begin()->buffer_size_;
      if((entry->totalSegs_ * max_sizePerFrame) > event_area_.capacity()) {
        event_area_.resize(entry->totalSegs_ * max_sizePerFrame);
      }
      unsigned char* pos = (unsigned char*)&event_area_[0];

      int sum=0;
      unsigned int lastpos=0;
      Fragments::iterator
        i(rc.first->second.begin()),e(rc.first->second.end());

      for(;i!=e;++i)
      {
        int dsize = i->buffer_size_;
        sum+=dsize;
        unsigned char* from=(unsigned char*)i->buffer_address_;
        copy(from,from+dsize,pos+lastpos);
        lastpos = lastpos + dsize;
        // ask deleter to kill off the buffer
        (*buffer_deleter_)(&(*i));
      }
      // the reformed DQM data is now in event_area_ deal with it
      DQMEventMsgView dqmEventView(&event_area_[0]);
      /* do not deserialize even for debug output but keep this for now
      // temporary debug output
      std::cout << "  DQM Message data:" << std::endl; 
      std::cout << "    protocol version = "
                << dqmEventView.protocolVersion() << std::endl; 
      std::cout << "    header size = "
                << dqmEventView.headerSize() << std::endl; 
      std::cout << "    run number = "
                << dqmEventView.runNumber() << std::endl; 
      std::cout << "    event number = "
                << dqmEventView.eventNumberAtUpdate() << std::endl; 
      std::cout << "    lumi section = "
                << dqmEventView.lumiSection() << std::endl; 
      std::cout << "    update number = "
                << dqmEventView.updateNumber() << std::endl; 
      std::cout << "    compression flag = "
                << dqmEventView.compressionFlag() << std::endl; 
      std::cout << "    reserved word = "
                << dqmEventView.reserved() << std::endl; 
      std::cout << "    release tag = "
                << dqmEventView.releaseTag() << std::endl; 
      std::cout << "    top folder name = "
                << dqmEventView.topFolderName() << std::endl; 
      std::cout << "    sub folder count = "
                << dqmEventView.subFolderCount() << std::endl; 
      edm::StreamDQMDeserializer deserializeWorker;
      std::auto_ptr<DQMEvent::TObjectTable> toTablePtr =
        deserializeWorker.deserializeDQMEvent(dqmEventView);
      DQMEvent::TObjectTable::const_iterator toIter;
      for (toIter = toTablePtr->begin();
           toIter != toTablePtr->end(); toIter++) {
        std::string subFolderName = toIter->first;
        std::cout << "  folder = " << subFolderName << std::endl;
        std::vector<TObject *> toList = toIter->second;
        for (int tdx = 0; tdx < (int) toList.size(); tdx++) {
          TObject *toPtr = toList[tdx];
          string cls = toPtr->IsA()->GetName();
          string nm = toPtr->GetName();
          std::cout << "    TObject class = " << cls
                    << ", name = " << nm << std::endl;
        }
      }
      */
      // Manage this DQMEvent, temporarily just stick it in the DQMEventServer
      //if (DQMeventServer_.get() != NULL)
      //{
      //  DQMeventServer_->processDQMEvent(dqmEventView);
      //}
      dqmServiceManager_->manageDQMEventMsg(dqmEventView);

      // remove the entry from the map
      fragment_area_.erase(rc.first);
    }
    ProgressMarker::instance()->processing(false);
  }
}

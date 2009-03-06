// $Id: FragmentCollector.cc,v 1.42.2.5 2008/11/15 20:02:49 biery Exp $

#include "EventFilter/StorageManager/interface/FragmentCollector.h"
#include "EventFilter/StorageManager/interface/ProgressMarker.h"


#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include "boost/bind.hpp"

#include "log4cplus/loggingmacros.h"

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
				       log4cplus::Logger& applicationLogger,
                                       const string& config_str):
    cmd_q_(&(h.getCommandQueue())),
    evtbuf_q_(&(h.getEventQueue())),
    frag_q_(&(h.getFragmentQueue())),
    buffer_deleter_(d),
    prods_(0),
    info_(&h), 
    lastStaleCheckTime_(time(0)),
    staleFragmentTimeout_(30),
    disks_(0),
    applicationLogger_(applicationLogger),
    writer_(new edm::ServiceManager(config_str)),
    dqmServiceManager_(new stor::DQMServiceManager())
  {
    // supposed to have given parameterSet smConfigString to writer_
    // at ctor
    event_area_.reserve(7000000);
  }
  FragmentCollector::FragmentCollector(std::auto_ptr<HLTInfo> info,Deleter d,
				       log4cplus::Logger& applicationLogger,
                                       const string& config_str):
    cmd_q_(&(info.get()->getCommandQueue())),
    evtbuf_q_(&(info.get()->getEventQueue())),
    frag_q_(&(info.get()->getFragmentQueue())),
    buffer_deleter_(d),
    prods_(0),
    info_(info.get()), 
    lastStaleCheckTime_(time(0)),
    staleFragmentTimeout_(30),
    disks_(0),
    applicationLogger_(applicationLogger),
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
    // 14-Oct-2008, KAB - avoid race condition by starting writers first
    // (otherwise INIT message could be received and processed before
    // the writers are started (and whatever initialization is done in the
    // writers when INIT messages are processed could be wiped out by 
    // the start command)
    writer_->start();
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
	  case Header::ERROR_EVENT:
	    {
	      FR_DEBUG << "FragColl: Got an Error_Event" << endl;
	      processErrorEvent(entry);
	      break;
	    }
	  case Header::FILE_CLOSE_REQUEST:
	    {
              FR_DEBUG << "FragColl: Got a File Close Request message" << endl;
              writer_->closeFilesIfNeeded();
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

    // verify that the segment number of the fragment is valid
    if (entry->segNumber_ < 1 || entry->segNumber_ > entry->totalSegs_)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Invalid fragment ID received for event " << entry->id_
                      << " in run " << entry->run_ << " with output module ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }

    // add a new entry to the fragment area (Collection) based on this
    // fragment's key (or fetch the existing entry if a fragment with the
    // same key has already been processed)
    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(FragKey(entry->code_, entry->run_, entry->id_,
                                              entry->secondaryId_, entry->originatorPid_,
                                              entry->originatorGuid_),
                                      FragmentContainer()));

    // add this fragment to the map of fragments for this event
    // (fragment map has key/value of fragment/segment ID and FragEntry)
    FragmentContainer& fragContainer = rc.first->second;
    std::map<int, FragEntry>& eventFragmentMap = fragContainer.fragmentMap_;
    pair<std::map<int, FragEntry>::iterator, bool> fragInsertResult =
      eventFragmentMap.insert(make_pair(entry->segNumber_, *entry));
    bool duplicateEntry = ! fragInsertResult.second;

    // if the specified fragment/segment ID already existed in the
    // map, complain and clean up
    if (duplicateEntry)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Duplicate fragment ID received for event " << entry->id_
                      << " in run " << entry->run_ << " with output module ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }
    // otherwise, we update the last fragment time for this event
    else {
      fragContainer.lastFragmentTime_ = time(0);
    }

    FR_DEBUG << "FragColl: added fragment with segment number "
             << entry->segNumber_ << endl;

    if((int)eventFragmentMap.size()==entry->totalSegs_)
    {
	FR_DEBUG << "FragColl: completed an event with "
		 << entry->totalSegs_ << " segments" << endl;

        // the assembleFragments method has several side-effects:
        // - the event_area_ is filled, and it may be resized
        // - the fragment entries are deleted using the buffer_deleter_
        int assembledSize = assembleFragments(eventFragmentMap);

        EventMsgView emsg(&event_area_[0]);
        FR_DEBUG << "FragColl: writing event size " << assembledSize << endl;
        writer_->manageEventMsg(emsg);

        if (eventServer_.get() != NULL)
        {
          eventServer_->processEvent(emsg);
        }

	// remove the entry from the map
	fragment_area_.erase(rc.first);

        // check for stale fragments
        removeStaleFragments();
    }
    ProgressMarker::instance()->processing(false);
  }

  void FragmentCollector::processHeader(FragEntry* entry)
  {
    ProgressMarker::instance()->processing(true);
    if(entry->totalSegs_==1)
    {
      FR_DEBUG << "FragColl: Got an INIT message with one segment" << endl;
      FR_DEBUG << "FragColl: Output Module ID " << entry->secondaryId_ << endl;

      // send immediately
      InitMsgView msg(entry->buffer_address_);
      FR_DEBUG << "FragColl: writing INIT size " << entry->buffer_size_ << endl;
      writer_->manageInitMsg(catalog_, disks_, sourceId_, msg, *initMsgCollection_);

      try
      {
        if (initMsgCollection_->addIfUnique(msg))
        {
          // check if any currently connected consumers did not specify
          // an HLT output module label and we now have multiple, different,
          // INIT messages.  If so, we need to complain because the
          // SelectHLTOutput parameter needs to be specified when there
          // is more than one HLT output module (and correspondingly, more
          // than one INIT message)
          if (initMsgCollection_->size() > 1)
          {
            std::map< uint32, boost::shared_ptr<ConsumerPipe> > consumerTable = 
              eventServer_->getConsumerTable();
            std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator 
              consumerIter;
            for (consumerIter = consumerTable.begin();
                 consumerIter != consumerTable.end();
                 ++consumerIter)
            {
              boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;

              // for regular consumers, we need to test whether the consumer
              // configuration specified an HLT output module
              if (! consPtr->isProxyServer())
              {
                if (consPtr->getHLTOutputSelection().empty())
                {
                  // store a warning message in the consumer pipe to be
                  // sent to the consumer at the next opportunity
                  std::string errorString;
                  errorString.append("ERROR: The configuration for this ");
                  errorString.append("consumer does not specify an HLT output ");
                  errorString.append("module.\nPlease specify one of the HLT ");
                  errorString.append("output modules listed below as the ");
                  errorString.append("SelectHLTOutput parameter ");
                  errorString.append("in the InputSource configuration.\n");
                  errorString.append(initMsgCollection_->getSelectionHelpString());
                  errorString.append("\n");
                  consPtr->setRegistryWarning(errorString);
                }
              }
            }
          }
        }
      }
      catch(cms::Exception& excpt)
      {
        char tidString[32];
        sprintf(tidString, "%d", entry->hltTid_);
        std::string logMsg = "receiveRegistryMessage: Error processing a ";
        logMsg.append("registry message from URL ");
        logMsg.append(entry->hltURL_);
        logMsg.append(" and Tid ");
        logMsg.append(tidString);
        logMsg.append(":\n");
        logMsg.append(excpt.what());
        logMsg.append("\n");
        logMsg.append(initMsgCollection_->getSelectionHelpString());
        FDEBUG(9) << logMsg << std::endl;
        LOG4CPLUS_ERROR(applicationLogger_, logMsg);

        throw excpt;
      }

      smRBSenderList_->registerDataSender(&entry->hltURL_[0], &entry->hltClassName_[0],
                                          entry->hltLocalId_, entry->hltInstance_, entry->hltTid_,
                                          entry->segNumber_, entry->totalSegs_, msg.size(),
                                          msg.outputModuleLabel(), msg.outputModuleId(),
                                          entry->rbBufferID_);

      // make sure the buffer properly released
      (*buffer_deleter_)(entry);
      return;
    } // end of single segment test

    // verify that the segment number of the fragment is valid
    if (entry->segNumber_ < 1 || entry->segNumber_ > entry->totalSegs_)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Invalid fragment ID received for INIT " << entry->id_
                      << " in run " << entry->run_ << " with output module ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }

    // add a new entry to the fragment area (Collection) based on this
    // fragment's key (or fetch the existing entry if a fragment with the
    // same key has already been processed)
    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(FragKey(entry->code_, entry->run_, entry->id_,
                                              entry->secondaryId_, entry->originatorPid_,
                                              entry->originatorGuid_),
                                      FragmentContainer()));

    // add this fragment to the map of fragments for this event
    // (fragment map has key/value of fragment/segment ID and FragEntry)
    FragmentContainer& fragContainer = rc.first->second;
    std::map<int, FragEntry>& eventFragmentMap = fragContainer.fragmentMap_;
    pair<std::map<int, FragEntry>::iterator, bool> fragInsertResult =
      eventFragmentMap.insert(make_pair(entry->segNumber_, *entry));
    bool duplicateEntry = ! fragInsertResult.second;

    // if the specified fragment/segment ID already existed in the
    // map, complain and clean up
    if (duplicateEntry)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Duplicate fragment ID received for INIT " << entry->id_
                      << " in run " << entry->run_ << " with output module ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }
    // otherwise, we update the last fragment time for this event
    else {
      fragContainer.lastFragmentTime_ = time(0);
    }

    FR_DEBUG << "FragColl: added INIT fragment with segment number "
             << entry->segNumber_ << endl;

    if((int)eventFragmentMap.size()==entry->totalSegs_)
    {
      FR_DEBUG << "FragColl: completed an INIT message with "
               << entry->totalSegs_ << " segments" << endl;

      // the assembleFragments method has several side-effects:
      // - the event_area_ is filled, and it may be resized
      // - the fragment entries are deleted using the buffer_deleter_
      int assembledSize = assembleFragments(eventFragmentMap);

      InitMsgView msg(&event_area_[0]);
      FR_DEBUG << "FragColl: writing INIT size " << assembledSize << endl;
      writer_->manageInitMsg(catalog_, disks_, sourceId_, msg, *initMsgCollection_);

      try
      {
        if (initMsgCollection_->addIfUnique(msg))
        {
          // check if any currently connected consumers did not specify
          // an HLT output module label and we now have multiple, different,
          // INIT messages.  If so, we need to complain because the
          // SelectHLTOutput parameter needs to be specified when there
          // is more than one HLT output module (and correspondingly, more
          // than one INIT message)
          if (initMsgCollection_->size() > 1)
          {
            std::map< uint32, boost::shared_ptr<ConsumerPipe> > consumerTable = 
              eventServer_->getConsumerTable();
            std::map< uint32, boost::shared_ptr<ConsumerPipe> >::const_iterator 
              consumerIter;
            for (consumerIter = consumerTable.begin();
                 consumerIter != consumerTable.end();
                 ++consumerIter)
            {
              boost::shared_ptr<ConsumerPipe> consPtr = consumerIter->second;

              // for regular consumers, we need to test whether the consumer
              // configuration specified an HLT output module
              if (! consPtr->isProxyServer())
              {
                if (consPtr->getHLTOutputSelection().empty())
                {
                  // store a warning message in the consumer pipe to be
                  // sent to the consumer at the next opportunity
                  std::string errorString;
                  errorString.append("ERROR: The configuration for this ");
                  errorString.append("consumer does not specify an HLT output ");
                  errorString.append("module.\nPlease specify one of the HLT ");
                  errorString.append("output modules listed below as the ");
                  errorString.append("SelectHLTOutput parameter ");
                  errorString.append("in the InputSource configuration.\n");
                  errorString.append(initMsgCollection_->getSelectionHelpString());
                  errorString.append("\n");
                  consPtr->setRegistryWarning(errorString);
                }
              }
            }
          }
        }
      }
      catch(cms::Exception& excpt)
      {
        char tidString[32];
        sprintf(tidString, "%d", entry->hltTid_);
        std::string logMsg = "receiveRegistryMessage: Error processing a ";
        logMsg.append("registry message from URL ");
        logMsg.append(entry->hltURL_);
        logMsg.append(" and Tid ");
        logMsg.append(tidString);
        logMsg.append(":\n");
        logMsg.append(excpt.what());
        logMsg.append("\n");
        logMsg.append(initMsgCollection_->getSelectionHelpString());
        FDEBUG(9) << logMsg << std::endl;
        LOG4CPLUS_ERROR(applicationLogger_, logMsg);

        throw excpt;
      }

      smRBSenderList_->registerDataSender(&entry->hltURL_[0], &entry->hltClassName_[0],
                                          entry->hltLocalId_, entry->hltInstance_, entry->hltTid_,
                                          entry->segNumber_, entry->totalSegs_, msg.size(),
                                          msg.outputModuleLabel(), msg.outputModuleId(),
                                          entry->rbBufferID_);

      // remove the entry from the map
      fragment_area_.erase(rc.first);

      // check for stale fragments
      removeStaleFragments();
    }
    ProgressMarker::instance()->processing(false);
  }

  void FragmentCollector::processDQMEvent(FragEntry* entry)
  {
    ProgressMarker::instance()->processing(true);
    if(entry->totalSegs_==1)
    {
      FR_DEBUG << "FragColl: Got a DQM_Event with one segment" << endl;
      FR_DEBUG << "FragColl: DQM_Event size " << entry->buffer_size_ << endl;
      FR_DEBUG << "FragColl: DQM_Event ID " << entry->id_ << endl;
      FR_DEBUG << "FragColl: DQM_Event folderID " << entry->secondaryId_ << endl;

      DQMEventMsgView dqmEventView(entry->buffer_address_);
      dqmServiceManager_->manageDQMEventMsg(dqmEventView);
      (*buffer_deleter_)(entry);
      return;
    } // end of single segment test

    // verify that the segment number of the fragment is valid
    if (entry->segNumber_ < 1 || entry->segNumber_ > entry->totalSegs_)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Invalid fragment ID received for DQM event " << entry->id_
                      << " in run " << entry->run_ << " with folder ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }

    // add a new entry to the fragment area (Collection) based on this
    // fragment's key (or fetch the existing entry if a fragment with the
    // same key has already been processed)
    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(FragKey(entry->code_, entry->run_, entry->id_,
                                              entry->secondaryId_, entry->originatorPid_,
                                              entry->originatorGuid_),
                                      FragmentContainer()));

    // add this fragment to the map of fragments for this event
    // (fragment map has key/value of fragment/segment ID and FragEntry)
    FragmentContainer& fragContainer = rc.first->second;
    std::map<int, FragEntry>& eventFragmentMap = fragContainer.fragmentMap_;
    pair<std::map<int, FragEntry>::iterator, bool> fragInsertResult =
      eventFragmentMap.insert(make_pair(entry->segNumber_, *entry));
    bool duplicateEntry = ! fragInsertResult.second;

    // if the specified fragment/segment ID already existed in the
    // map, complain and clean up
    if (duplicateEntry)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Duplicate fragment ID received for DQM event " << entry->id_
                      << " in run " << entry->run_ << " with folder ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }
    // otherwise, we update the last fragment time for this event
    else {
      fragContainer.lastFragmentTime_ = time(0);
    }

    FR_DEBUG << "FragColl: added DQM fragment" << endl;
    
    if((int)eventFragmentMap.size()==entry->totalSegs_)
    {
      FR_DEBUG << "FragColl: completed a DQM_event with "
       << entry->totalSegs_ << " segments" << endl;

      // the assembleFragments method has several side-effects:
      // - the event_area_ is filled, and it may be resized
      // - the fragment entries are deleted using the buffer_deleter_
      assembleFragments(eventFragmentMap);

      // the reformed DQM data is now in event_area_ deal with it
      DQMEventMsgView dqmEventView(&event_area_[0]);
      dqmServiceManager_->manageDQMEventMsg(dqmEventView);

      // remove the entry from the map
      fragment_area_.erase(rc.first);

      // check for stale fragments
      removeStaleFragments();
    }
    ProgressMarker::instance()->processing(false);
  }

  void FragmentCollector::processErrorEvent(FragEntry* entry)
  {
    ProgressMarker::instance()->processing(true);
    if(entry->totalSegs_==1)
    {
	FR_DEBUG << "FragColl: Got an Error Event with one segment" << endl;
	FR_DEBUG << "FragColl: Event size " << entry->buffer_size_ << endl;
	FR_DEBUG << "FragColl: Event ID " << entry->id_ << endl;

	// send immediately
        FRDEventMsgView emsg(entry->buffer_address_);
        FR_DEBUG << "FragColl: writing error event size " << entry->buffer_size_ << endl;
        writer_->manageErrorEventMsg(catalog_, disks_, sourceId_, emsg);

	// make sure the buffer properly released
	(*buffer_deleter_)(entry);
	return;
    } // end of single segment test

    // verify that the segment number of the fragment is valid
    if (entry->segNumber_ < 1 || entry->segNumber_ > entry->totalSegs_)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Invalid fragment ID received for Error event " << entry->id_
                      << " in run " << entry->run_ << " with output module ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }

    // add a new entry to the fragment area (Collection) based on this
    // fragment's key (or fetch the existing entry if a fragment with the
    // same key has already been processed)
    pair<Collection::iterator,bool> rc =
      fragment_area_.insert(make_pair(FragKey(entry->code_, entry->run_, entry->id_,
                                              entry->secondaryId_, entry->originatorPid_,
                                              entry->originatorGuid_),
                                      FragmentContainer()));

    // add this fragment to the map of fragments for this event
    // (fragment map has key/value of fragment/segment ID and FragEntry)
    FragmentContainer& fragContainer = rc.first->second;
    std::map<int, FragEntry>& eventFragmentMap = fragContainer.fragmentMap_;
    pair<std::map<int, FragEntry>::iterator, bool> fragInsertResult =
      eventFragmentMap.insert(make_pair(entry->segNumber_, *entry));
    bool duplicateEntry = ! fragInsertResult.second;

    // if the specified fragment/segment ID already existed in the
    // map, complain and clean up
    if (duplicateEntry)
    {
      LOG4CPLUS_ERROR(applicationLogger_,
                      "Duplicate fragment ID received for Error event " << entry->id_
                      << " in run " << entry->run_ << " with output module ID of "
                      << entry->secondaryId_ << ", FU PID = "
                      << entry->originatorPid_ << ", FU GUID = "
                      << entry->originatorGuid_  << ".  Fragment id is "
                      << entry->segNumber_ << ", total number of fragments is "
                      << entry->totalSegs_ << ".");
      (*buffer_deleter_)(entry);
      return;
    }
    // otherwise, we update the last fragment time for this event
    else {
      fragContainer.lastFragmentTime_ = time(0);
    }

    FR_DEBUG << "FragColl: added Error event fragment" << endl;
    
    if((int)eventFragmentMap.size()==entry->totalSegs_)
    {
	FR_DEBUG << "FragColl: completed an error event with "
		 << entry->totalSegs_ << " segments" << endl;

        // the assembleFragments method has several side-effects:
        // - the event_area_ is filled, and it may be resized
        // - the fragment entries are deleted using the buffer_deleter_
        int assembledSize = assembleFragments(eventFragmentMap);

        FRDEventMsgView emsg(&event_area_[0]);
        FR_DEBUG << "FragColl: writing error event size " << assembledSize << endl;
        writer_->manageErrorEventMsg(catalog_, disks_, sourceId_, emsg);

	// remove the entry from the map
	fragment_area_.erase(rc.first);

        // check for stale fragments
        removeStaleFragments();
    }
    ProgressMarker::instance()->processing(false);
  }

  /**
   * This method copies the data from the fragments contained in the
   * specified fragmentMap to the event_area_ attribute of this class.
   * The fragment map is keyed by the fragment number, where the fragment
   * number runs from 1 to N (the number of fragments). 
   * This method has several side-effects:  it changes the contents of
   * the event_area_ attribute, it may resize the event_area_ attribute if
   * it needs to be made larger (to handle all of the fragments, and
   * the fragment buffers are deleted using the buffer_deleter_ attribute
   * of this class.
   *
   * @return the number of bytes copied into the event_area_.
   */
  int FragmentCollector::assembleFragments(std::map<int, FragEntry>& fragmentMap)
  {
    // first make sure we have enough room; use an overestimate
    unsigned int max_sizePerFrame = fragmentMap[1].buffer_size_;
    if((fragmentMap.size() * max_sizePerFrame) > event_area_.capacity())
    {
      event_area_.resize(fragmentMap.size() * max_sizePerFrame);
    }
    unsigned char* pos = (unsigned char*)&event_area_[0];
	
    int sum=0;
    unsigned int lastpos=0;
    for (unsigned int idx = 1; idx <= fragmentMap.size(); ++idx)
    {
      FragEntry& workingEntry = fragmentMap[idx];
      int dsize = workingEntry.buffer_size_;
      sum+=dsize;
      unsigned char* from=(unsigned char*)workingEntry.buffer_address_;
      copy(from,from+dsize,pos+lastpos);
      lastpos = lastpos + dsize;
      // ask deleter to kill off the buffer
      (*buffer_deleter_)(&workingEntry);
    }

    return sum;
  }

  /**
   * This method removes stale fragments from the fragment_area_ (Collection).
   * Obviously, it has the side-effect of modifying the fragment_area_
   *
   * @return the number of events (fragmentContainers, actually) that
   *         were removed from the fragment_area_.
   */
  int FragmentCollector::removeStaleFragments()
  {
    // if there are no entries in the fragment_area_, we know
    // right away that there are no stale fragments
    if (fragment_area_.size() == 0) {return 0;}

    // check if it is time to look for stale fragments
    // (we could have a separate interval specified for how often
    // to run the test, but for now, we'll just use an interval
    // of the stale timeout.  So, the staleFragmentTimeout is doing
    // double duty - it tells us how old fragments must be before
    // we delete them and it tells us how often to run the test of
    // whether any stale fragments exist.
    time_t now = time(0);
    if ((now - lastStaleCheckTime_) < staleFragmentTimeout_) {return 0;}

    lastStaleCheckTime_ = now;
    //LOG4CPLUS_INFO(applicationLogger_,
    //               "Running the stale fragment test at " << now
    //               << ", number of entries in the fragment area is "
    //               << fragment_area_.size() << ".");

    //  build up a list of events that need to be removed
    std::vector<FragKey> staleList;
    Collection::iterator areaIter;
    for (areaIter = fragment_area_.begin();
         areaIter != fragment_area_.end();
         ++areaIter)
    {
      FragmentContainer& fragContainer = areaIter->second;
      std::map<int, FragEntry>::iterator fragIter =
        fragContainer.fragmentMap_.begin();
      FragEntry& workingEntry = fragIter->second;
      //LOG4CPLUS_INFO(applicationLogger_,
      //               "Testing if the fragments for event " << workingEntry.id_
      //               << " in run " << workingEntry.run_ << " are stale.  "
      //               << "Now = " << now << ", lastFragmentTime = "
      //               << fragContainer.lastFragmentTime_ << ".");
      // remember, the granularity of the times is a (large) one second
      if (fragContainer.lastFragmentTime_ > 0 &&
          fragContainer.lastFragmentTime_ >= fragContainer.creationTime_ &&
          (now - fragContainer.lastFragmentTime_) > staleFragmentTimeout_)
      {
        LOG4CPLUS_WARN(applicationLogger_,
                       "Deleting a stale fragment set for event "
                       << workingEntry.id_ << " in run " << workingEntry.run_
                       << " with secondary ID of " << workingEntry.secondaryId_
                       << ", FU PID = " << workingEntry.originatorPid_
                       << ", FU GUID = " << workingEntry.originatorGuid_
                       << ".  The number of fragments received was "
                       << fragContainer.fragmentMap_.size()
                       << ", and the total number of fragments expected was "
                       << workingEntry.totalSegs_ << ".");
        staleList.push_back(areaIter->first);
      }
    }

    // actually do the removal
    for (unsigned int idx = 0; idx < staleList.size(); ++idx)
    {
      fragment_area_.erase(staleList[idx]);
    }

    return staleList.size();
  }
}

// $Id: EventMsgData.cc,v 1.11 2011/03/08 18:34:11 mommsen Exp $
/// @file: EventMsgData.cc

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/EventMessage.h"

#include <stdlib.h>

namespace stor
{

  namespace detail
  {

    EventMsgData::EventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(I2O_SM_DATA, Header::EVENT),
      headerFieldsCached_(false)
    {
      addFirstFragment(pRef);
      parseI2OHeader();
    }

    inline size_t EventMsgData::do_i2oFrameSize() const
    {
      return sizeof(I2O_SM_DATA_MESSAGE_FRAME);
    }

    unsigned long EventMsgData::do_headerSize() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerSize_;
    }

    unsigned char* EventMsgData::do_headerLocation() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerLocation_;
    }

    inline unsigned char*
    EventMsgData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      if ( parsable() )
      {
        I2O_SM_DATA_MESSAGE_FRAME *smMsg =
          (I2O_SM_DATA_MESSAGE_FRAME*) dataLoc;
        return (unsigned char*) smMsg->dataPtr();
      }
      else
      {
        return dataLoc;
      }
    }

    uint32_t EventMsgData::do_outputModuleId() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An output module ID can not be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return outputModuleId_;
    }

    uint32_t EventMsgData::do_hltTriggerCount() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The number of HLT trigger bits can not be determined ";
        msg << "from a faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return hltTriggerCount_;
    }

    void
    EventMsgData::do_hltTriggerBits(std::vector<unsigned char>& bitList) const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The HLT trigger bits can not be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      bitList = hltTriggerBits_;
    }

    unsigned int
    EventMsgData::do_droppedEventsCount() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The dropped events count cannot be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return droppedEventsCount_;
     }

    void
    EventMsgData::do_setDroppedEventsCount(unsigned int count)
    {
      if ( headerOkay() )
      {
        const unsigned long firstFragSize = dataSize(0);
        
        // This should always be the case:
        assert( firstFragSize > sizeof(EventHeader) );

        EventHeader* header = (EventHeader*)dataLocation(0);
        convert(count,header->droppedEventsCount_);
      }
    }

    void 
    EventMsgData::do_assertRunNumber(uint32_t runNumber)
    {
      if ( headerOkay() && do_runNumber() != runNumber )
      {
        std::ostringstream errorMsg;
        errorMsg << "Run number " << do_runNumber() 
          << " of event " << do_eventNumber() <<
          " received from " << hltURL() <<
          " (FU process id " << fuProcessId() << ")" <<
          " does not match the run number " << runNumber << 
          " used to configure the StorageManager.";
        XCEPT_RAISE(stor::exception::RunNumberMismatch, errorMsg.str());
      }
    }

    uint32_t EventMsgData::do_runNumber() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "A run number can not be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return runNumber_;
    }

    uint32_t EventMsgData::do_lumiSection() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "A luminosity section can not be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return lumiSection_;
    }

    uint32_t EventMsgData::do_eventNumber() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An event number can not be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return eventNumber_;
    }

    uint32_t EventMsgData::do_adler32Checksum() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An adler32 checksum can not be determined from a ";
        msg << "faulty or incomplete Event message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return adler32_;
    }

    inline void EventMsgData::parseI2OHeader()
    {
      if ( parsable() )
      {
        I2O_SM_DATA_MESSAGE_FRAME *smMsg =
          (I2O_SM_DATA_MESSAGE_FRAME*) ref_->getDataLocation();
        fragKey_.code_ = messageCode_;
        fragKey_.run_ = smMsg->runID;
        fragKey_.event_ = smMsg->eventID;
        fragKey_.secondaryId_ = smMsg->outModID;
        fragKey_.originatorPid_ = smMsg->fuProcID;
        fragKey_.originatorGuid_ = smMsg->fuGUID;
        rbBufferId_ = smMsg->rbBufferID;
        hltLocalId_ = smMsg->hltLocalId;
        hltInstance_ = smMsg->hltInstance;
        hltTid_ = smMsg->hltTid;
        fuProcessId_ = smMsg->fuProcID;
        fuGuid_ = smMsg->fuGUID;
      }
    }

    void EventMsgData::cacheHeaderFields() const
    {
      unsigned char* firstFragLoc = dataLocation(0);
      unsigned long firstFragSize = dataSize(0);
      bool useFirstFrag = false;

      // if there is only one fragment, use it
      if (fragmentCount_ == 1)
      {
        useFirstFrag = true;
      }
      // otherwise, check if the first fragment is large enough to hold
      // the full Event message header  (we require some minimal fixed
      // size in the hope that we don't parse garbage when we overlay
      // the EventMsgView on the buffer)
      else if (firstFragSize > (sizeof(EventHeader) + 4096))
      {
        EventMsgView view(firstFragLoc);
        if (view.headerSize() <= firstFragSize)
        {
          useFirstFrag = true;
        }
      }

      boost::shared_ptr<EventMsgView> msgView;
      if (useFirstFrag)
      {
        msgView.reset(new EventMsgView(firstFragLoc));
      }
      else
      {
        copyFragmentsIntoBuffer(headerCopy_);
        msgView.reset(new EventMsgView(&headerCopy_[0]));
      }

      headerSize_ = msgView->headerSize();
      headerLocation_ = msgView->startAddress();
      outputModuleId_ = msgView->outModId();
      hltTriggerCount_ = msgView->hltCount();
      if (hltTriggerCount_ > 0)
        {
          hltTriggerBits_.resize(1 + (hltTriggerCount_-1)/4);
        }
      msgView->hltTriggerBits(&hltTriggerBits_[0]);

      runNumber_ = msgView->run();
      lumiSection_ = msgView->lumi();
      eventNumber_ = msgView->event();
      adler32_ = msgView->adler32_chksum();
      droppedEventsCount_ = msgView->droppedEventsCount();

      headerFieldsCached_ = true;

      #ifdef STOR_DEBUG_WRONG_ADLER
      double r = rand()/static_cast<double>(RAND_MAX);
      if (r < 0.01)
      {
        std::cout << "Simulating corrupt Adler calculation" << std::endl;
        headerSize_ += 3;
      }
      else if (r < 0.02)
      {
        std::cout << "Simulating corrupt Adler entry" << std::endl;
        adler32_ += r*10000;
      }
      #endif // STOR_DEBUG_WRONG_ADLER
    }

  } // namespace detail

} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

// $Id: EventMsgData.cc,v 1.7 2010/05/17 15:59:10 mommsen Exp $
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
      _headerFieldsCached(false)
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerSize;
    }

    unsigned char* EventMsgData::do_headerLocation() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerLocation;
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _outputModuleId;
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _hltTriggerCount;
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      bitList = _hltTriggerBits;
    }

    void
    EventMsgData::do_addDroppedEventsCount(unsigned int count)
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _runNumber;
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _lumiSection;
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _eventNumber;
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

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _adler32;
    }

    inline void EventMsgData::parseI2OHeader()
    {
      if ( parsable() )
      {
        I2O_SM_DATA_MESSAGE_FRAME *smMsg =
          (I2O_SM_DATA_MESSAGE_FRAME*) _ref->getDataLocation();
        _fragKey.code_ = _messageCode;
        _fragKey.run_ = smMsg->runID;
        _fragKey.event_ = smMsg->eventID;
        _fragKey.secondaryId_ = smMsg->outModID;
        _fragKey.originatorPid_ = smMsg->fuProcID;
        _fragKey.originatorGuid_ = smMsg->fuGUID;
        _rbBufferId = smMsg->rbBufferID;
        _hltLocalId = smMsg->hltLocalId;
        _hltInstance = smMsg->hltInstance;
        _hltTid = smMsg->hltTid;
        _fuProcessId = smMsg->fuProcID;
        _fuGuid = smMsg->fuGUID;
      }
    }

    void EventMsgData::cacheHeaderFields() const
    {
      unsigned char* firstFragLoc = dataLocation(0);
      unsigned long firstFragSize = dataSize(0);
      bool useFirstFrag = false;

      // if there is only one fragment, use it
      if (_fragmentCount == 1)
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
        copyFragmentsIntoBuffer(_headerCopy);
        msgView.reset(new EventMsgView(&_headerCopy[0]));
      }

      _headerSize = msgView->headerSize();
      _headerLocation = msgView->startAddress();
      _outputModuleId = msgView->outModId();
      _hltTriggerCount = msgView->hltCount();
      if (_hltTriggerCount > 0)
        {
          _hltTriggerBits.resize(1 + (_hltTriggerCount-1)/4);
        }
      msgView->hltTriggerBits(&_hltTriggerBits[0]);

      _runNumber = msgView->run();
      _lumiSection = msgView->lumi();
      _eventNumber = msgView->event();
      _adler32 = msgView->adler32_chksum();

      _headerFieldsCached = true;

      #ifdef STOR_DEBUG_WRONG_ADLER
      double r = rand()/static_cast<double>(RAND_MAX);
      if (r < 0.01)
      {
        std::cout << "Simulating corrupt Adler calculation" << std::endl;
        _headerSize += 3;
      }
      else if (r < 0.02)
      {
        std::cout << "Simulating corrupt Adler entry" << std::endl;
        _adler32 += r*10000;
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

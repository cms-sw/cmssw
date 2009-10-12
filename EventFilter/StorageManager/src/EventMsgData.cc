// $Id: $

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/EventMessage.h"

namespace stor
{

  namespace detail
  {

    EventMsgData::EventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(pRef),
      _headerFieldsCached(false)
    {
      parseI2OHeader();

      if (_fragmentCount > 1)
        {
          toolbox::mem::Reference* curRef = _ref->getNextReference();
          while (curRef)
            {
              validateMessageCode(curRef, I2O_SM_DATA);
              curRef = curRef->getNextReference();
            }
        }

      if (!faulty() && _fragmentCount == _expectedNumberOfFragments)
        {
          markComplete();
        }
    }

    unsigned long EventMsgData::do_headerSize() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerSize;
    }

    unsigned char* EventMsgData::do_headerLocation() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerLocation;
    }

    inline unsigned char*
    EventMsgData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      if (parsable())
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

    uint32 EventMsgData::do_outputModuleId() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "An output module ID can not be determined from a ";
          msg << "faulty or incomplete Event message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _outputModuleId;
    }

    uint32 EventMsgData::do_hltTriggerCount() const
    {
      if (faulty() || !complete())
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
      if (faulty() || !complete())
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
    EventMsgData::do_assertRunNumber(uint32 runNumber)
    {
      if ( do_runNumber() != runNumber )
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

    uint32 EventMsgData::do_runNumber() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "A run number can not be determined from a ";
          msg << "faulty or incomplete Event message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _runNumber;
    }

    uint32 EventMsgData::do_lumiSection() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "A luminosity section can not be determined from a ";
          msg << "faulty or incomplete Event message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _lumiSection;
    }

    uint32 EventMsgData::do_eventNumber() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "An event number can not be determined from a ";
          msg << "faulty or incomplete Event message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _eventNumber;
    }

    inline void EventMsgData::parseI2OHeader()
    {
      if (parsable())
        {
          _messageCode = Header::EVENT;
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

      _headerFieldsCached = true;
    }

  } // namespace detail

} // namespace stor

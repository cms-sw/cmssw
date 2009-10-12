// $Id: $

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/FRDEventMessage.h"

namespace stor
{

  namespace detail
  {

    ErrorEventMsgData::ErrorEventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(pRef),
      _headerFieldsCached(false)
    {
      parseI2OHeader();

      if (_fragmentCount > 1)
        {
          toolbox::mem::Reference* curRef = _ref->getNextReference();
          while (curRef)
            {
              validateMessageCode(curRef, I2O_SM_ERROR);
              curRef = curRef->getNextReference();
            }
        }

      if (!faulty() && _fragmentCount == _expectedNumberOfFragments)
        {
          markComplete();
        }
    }

    unsigned long ErrorEventMsgData::do_headerSize() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerSize;
    }

    unsigned char* ErrorEventMsgData::do_headerLocation() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerLocation;
    }

    inline unsigned char*
    ErrorEventMsgData::do_fragmentLocation(unsigned char* dataLoc) const
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

    void
    ErrorEventMsgData::do_assertRunNumber(uint32 runNumber)
    {
      if ( do_runNumber() != runNumber )
      {
        std::ostringstream errorMsg;
        errorMsg << "Run number " << do_runNumber() 
          << " of error event " << do_eventNumber() <<
          " received from " << hltURL() << 
          " (FU process id " << fuProcessId() << ")" <<
          " does not match the run number " << runNumber << 
          " used to configure the StorageManager." <<
          " Enforce usage of configured run number.";
        _runNumber = runNumber;
        XCEPT_RAISE(stor::exception::RunNumberMismatch, errorMsg.str());
      }
    }

    uint32 ErrorEventMsgData::do_runNumber() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "A run number can not be determined from a ";
          msg << "faulty or incomplete ErrorEvent message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _runNumber;
    }

    uint32 ErrorEventMsgData::do_lumiSection() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "A luminosity section can not be determined from a ";
          msg << "faulty or incomplete ErrorEvent message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _lumiSection;
    }

    uint32 ErrorEventMsgData::do_eventNumber() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "An event number can not be determined from a ";
          msg << "faulty or incomplete ErrorEvent message.";
          XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _eventNumber;
    }

    inline void ErrorEventMsgData::parseI2OHeader()
    {
      if (parsable())
        {
          _messageCode = Header::ERROR_EVENT;
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

    void ErrorEventMsgData::cacheHeaderFields() const
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
      // the full Event message header  (FRD events have fixed header
      // size, so the check is easy)
      else if (firstFragSize > sizeof(FRDEventHeader_V2))
        {
          useFirstFrag = true;
        }

      boost::shared_ptr<FRDEventMsgView> msgView;
      std::vector<unsigned char> tempBuffer;
      if (useFirstFrag)
        {
          msgView.reset(new FRDEventMsgView(firstFragLoc));
        }
      else
        {
          copyFragmentsIntoBuffer(tempBuffer);
          msgView.reset(new FRDEventMsgView(&tempBuffer[0]));
        }

      _headerSize = sizeof(FRDEventHeader_V2);
      _headerLocation = msgView->startAddress();

      _runNumber = msgView->run();
      _lumiSection = msgView->lumi();
      _eventNumber = msgView->event();

      _headerFieldsCached = true;
    }

  } // namespace detail

} // namespace stor

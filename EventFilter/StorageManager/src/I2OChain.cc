// $Id: I2OChain.cc,v 1.8 2009/08/28 16:41:26 mommsen Exp $
/// @file: I2OChain.cc

#include <algorithm>

#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

#include "EventFilter/StorageManager/src/ChainData.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/MsgHeader.h"


namespace stor
{
  namespace detail
  {

    class InitMsgData : public ChainData
    {
    public:
      explicit InitMsgData(toolbox::mem::Reference* pRef);
      ~InitMsgData() {}

    protected:
      unsigned long do_headerSize() const;
      unsigned char* do_headerLocation() const;
      unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      uint32 do_outputModuleId() const;
      std::string do_outputModuleLabel() const;
      void do_hltTriggerNames(Strings& nameList) const;
      void do_hltTriggerSelections(Strings& nameList) const;
      void do_l1TriggerNames(Strings& nameList) const;

    private:
      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool _headerFieldsCached;
      mutable std::vector<unsigned char> _headerCopy;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable uint32 _outputModuleId;
      mutable std::string _outputModuleLabel;
      mutable Strings _hltTriggerNames;
      mutable Strings _hltTriggerSelections;
      mutable Strings _l1TriggerNames;
    };

    inline InitMsgData::InitMsgData(toolbox::mem::Reference* pRef) :
      ChainData(pRef),
      _headerFieldsCached(false)
    {
      parseI2OHeader();

      if (_fragmentCount > 1)
        {
          toolbox::mem::Reference* curRef = _ref->getNextReference();
          while (curRef)
            {
              validateMessageCode(curRef, I2O_SM_PREAMBLE);
              curRef = curRef->getNextReference();
            }
        }

      if (!faulty() && _fragmentCount == _expectedNumberOfFragments)
        {
          markComplete();
        }
    }

    unsigned long InitMsgData::do_headerSize() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerSize;
    }

    unsigned char* InitMsgData::do_headerLocation() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerLocation;
    }

    inline unsigned char*
    InitMsgData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      if (parsable())
        {
          I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
            (I2O_SM_PREAMBLE_MESSAGE_FRAME*) dataLoc;
          return (unsigned char*) smMsg->dataPtr();
        }
      else
        {
          return dataLoc;
        }
    }

    uint32 InitMsgData::do_outputModuleId() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "An output module ID can not be determined from a ";
          msg << "faulty or incomplete INIT message.";
          XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _outputModuleId;
    }

    std::string InitMsgData::do_outputModuleLabel() const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "An output module label can not be determined from a ";
          msg << "faulty or incomplete INIT message.";
          XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _outputModuleLabel;
    }

    void InitMsgData::do_hltTriggerNames(Strings& nameList) const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "The HLT trigger names can not be determined from a ";
          msg << "faulty or incomplete INIT message.";
          XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      nameList = _hltTriggerNames;
    }

    void InitMsgData::do_hltTriggerSelections(Strings& nameList) const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "The HLT trigger selections can not be determined from a ";
          msg << "faulty or incomplete INIT message.";
          XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      nameList = _hltTriggerSelections;
    }

    void InitMsgData::do_l1TriggerNames(Strings& nameList) const
    {
      if (faulty() || !complete())
        {
          std::stringstream msg;
          msg << "The L1 trigger names can not be determined from a ";
          msg << "faulty or incomplete INIT message.";
          XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      nameList = _l1TriggerNames;
    }

    inline void InitMsgData::parseI2OHeader()
    {
      if (parsable())
        {
          _messageCode = Header::INIT;
          I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
            (I2O_SM_PREAMBLE_MESSAGE_FRAME*) _ref->getDataLocation();
          _fragKey.code_ = _messageCode;
          _fragKey.run_ = 0;
          _fragKey.event_ = smMsg->hltTid;
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

    void InitMsgData::cacheHeaderFields() const
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
      // the full INIT message header  (we require some minimal fixed
      // size in the hope that we don't parse garbage when we overlay
      // the InitMsgView on the buffer)
      else if (firstFragSize > (sizeof(InitHeader) + 16384))
        {
          InitMsgView view(firstFragLoc);
          if (view.headerSize() <= firstFragSize)
            {
              useFirstFrag = true;
            }
        }

      boost::shared_ptr<InitMsgView> msgView;
      if (useFirstFrag)
        {
          msgView.reset(new InitMsgView(firstFragLoc));
        }
      else
        {
          copyFragmentsIntoBuffer(_headerCopy);
          msgView.reset(new InitMsgView(&_headerCopy[0]));
        }

      _headerSize = msgView->headerSize();
      _headerLocation = msgView->startAddress();
      _outputModuleId = msgView->outputModuleId();
      _outputModuleLabel = msgView->outputModuleLabel();
      msgView->hltTriggerNames(_hltTriggerNames);
      msgView->hltTriggerSelections(_hltTriggerSelections);
      msgView->l1TriggerNames(_l1TriggerNames);

      _headerFieldsCached = true;
    }

    class EventMsgData : public ChainData
    {
    public:
      explicit EventMsgData(toolbox::mem::Reference* pRef);
      ~EventMsgData() {}

    protected:
      unsigned long do_headerSize() const;
      unsigned char* do_headerLocation() const;
      unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      uint32 do_outputModuleId() const;
      uint32 do_hltTriggerCount() const;
      void do_hltTriggerBits(std::vector<unsigned char>& bitList) const;
      void do_assertRunNumber(uint32 runNumber);
      uint32 do_runNumber() const;
      uint32 do_lumiSection() const;
      uint32 do_eventNumber() const;

    private:
      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool _headerFieldsCached;
      mutable std::vector<unsigned char> _headerCopy;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable uint32 _outputModuleId;
      mutable uint32 _hltTriggerCount;
      mutable std::vector<unsigned char> _hltTriggerBits;
      mutable uint32 _runNumber;
      mutable uint32 _lumiSection;
      mutable uint32 _eventNumber;
    };

    inline EventMsgData::EventMsgData(toolbox::mem::Reference* pRef) :
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

    class DQMEventMsgData : public ChainData
    {
    public:
      explicit DQMEventMsgData(toolbox::mem::Reference* pRef);
      ~DQMEventMsgData() {}

    protected:
      unsigned long do_headerSize() const;
      unsigned char* do_headerLocation() const;
      unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      std::string do_topFolderName() const;
      DQMKey do_dqmKey() const;
      void do_assertRunNumber(uint32 runNumber);

    private:

      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool _headerFieldsCached;
      mutable std::vector<unsigned char> _headerCopy;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable std::string _topFolderName;
      mutable DQMKey _dqmKey;

    };

    inline DQMEventMsgData::DQMEventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(pRef)
    {

      _headerFieldsCached = false;

      parseI2OHeader();

      if (_fragmentCount > 1)
        {
          toolbox::mem::Reference* curRef = _ref->getNextReference();
          while (curRef)
            {
              validateMessageCode(curRef, I2O_SM_DQM);
              curRef = curRef->getNextReference();
            }
        }

      if (!faulty() && _fragmentCount == _expectedNumberOfFragments)
        {
          markComplete();
        }
    }

    inline std::string DQMEventMsgData::do_topFolderName() const
    {

      if( !_headerFieldsCached )
        {
          cacheHeaderFields();
        }

      if( faulty() || !complete() )
        {
          std::stringstream msg;
          msg << "A top folder name can not be determined from a ";
          msg << "faulty or incomplete DQM event message.";
          XCEPT_RAISE( stor::exception::IncompleteInitMessage, msg.str() );
        }

      return _topFolderName;

    }

    inline DQMKey DQMEventMsgData::do_dqmKey() const
    {

      if( !_headerFieldsCached )
        {
          cacheHeaderFields();
        }

      if( faulty() || !complete() )
        {
          std::stringstream msg;
          msg << "The DQM key can not be determined from a ";
          msg << "faulty or incomplete DQM event message.";
          XCEPT_RAISE( stor::exception::IncompleteInitMessage, msg.str() );
        }

      return _dqmKey;

    }

    void DQMEventMsgData::do_assertRunNumber(uint32 runNumber)
    {
      if ( do_dqmKey().runNumber != runNumber )
      {
        std::ostringstream errorMsg;
        errorMsg << "Run number " << do_runNumber() 
          << " of DQM event " << do_eventNumber() <<
          " received from " << hltURL() << 
          " (FU process id " << fuProcessId() << ")" <<
          " does not match the run number " << runNumber << 
          " used to configure the StorageManager.";
        XCEPT_RAISE(stor::exception::RunNumberMismatch, errorMsg.str());
      }
    }

    unsigned long DQMEventMsgData::do_headerSize() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerSize;
    }

    unsigned char* DQMEventMsgData::do_headerLocation() const
    {
      if (faulty() || !complete())
        {
          return 0;
        }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerLocation;
    }

    inline unsigned char*
    DQMEventMsgData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      if (parsable())
        {
          I2O_SM_DQM_MESSAGE_FRAME *smMsg =
            (I2O_SM_DQM_MESSAGE_FRAME*) dataLoc;
          return (unsigned char*) smMsg->dataPtr();
        }
      else
        {
          return dataLoc;
        }
    }

    inline void DQMEventMsgData::parseI2OHeader()
    {
      if (parsable())
        {
          _messageCode = Header::DQM_EVENT;
          I2O_SM_DQM_MESSAGE_FRAME *smMsg =
            (I2O_SM_DQM_MESSAGE_FRAME*) _ref->getDataLocation();
          _fragKey.code_ = _messageCode;
          _fragKey.run_ = smMsg->runID;
          _fragKey.event_ = smMsg->eventAtUpdateID;
          _fragKey.secondaryId_ = smMsg->folderID;
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

    // Adapted from InitMsgData::cacheHeaderFields
    void DQMEventMsgData::cacheHeaderFields() const
    {

      unsigned char* firstFragLoc = dataLocation(0);
      unsigned long firstFragSize = dataSize(0);
      bool useFirstFrag = false;

      if (_fragmentCount == 1)
        {
          useFirstFrag = true;
        }
      else if( firstFragSize > (sizeof(DQMEventHeader) + 8192) )
        {
          DQMEventMsgView view( firstFragLoc );
          if( view.headerSize() <= firstFragSize )
            {
              useFirstFrag = true;
            }
        }

      boost::shared_ptr<DQMEventMsgView> msgView;
      if (useFirstFrag)
        {
          msgView.reset(new DQMEventMsgView(firstFragLoc));
        }
      else
        {
          copyFragmentsIntoBuffer(_headerCopy);
          msgView.reset(new DQMEventMsgView(&_headerCopy[0]));
        }

      _headerSize = msgView->headerSize();
      _headerLocation = msgView->startAddress();
      _topFolderName = msgView->topFolderName();

      _dqmKey.runNumber = msgView->runNumber();
      _dqmKey.lumiSection = msgView->lumiSection();
      _dqmKey.updateNumber = msgView->updateNumber();

      _headerFieldsCached = true;

    }

    class ErrorEventMsgData : public ChainData
    {
    public:
      explicit ErrorEventMsgData(toolbox::mem::Reference* pRef);
      ~ErrorEventMsgData() {}

    protected:
      unsigned long do_headerSize() const;
      unsigned char* do_headerLocation() const;
      unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      void do_assertRunNumber(uint32 runNumber);
      uint32 do_runNumber() const;
      uint32 do_lumiSection() const;
      uint32 do_eventNumber() const;

    private:
      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool _headerFieldsCached;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable uint32 _runNumber;
      mutable uint32 _lumiSection;
      mutable uint32 _eventNumber;
    };

    inline ErrorEventMsgData::ErrorEventMsgData(toolbox::mem::Reference* pRef) :
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


  // A default-constructed I2OChain has a null (shared) pointer.
  I2OChain::I2OChain() :
    _data()
  { 
  }

  I2OChain::I2OChain(toolbox::mem::Reference* pRef)
  {
    if (pRef)
      {
        I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
          (I2O_PRIVATE_MESSAGE_FRAME*) pRef->getDataLocation();
        if (!pvtMsg || ((size_t)(pvtMsg->StdMessageFrame.MessageSize*4) <
                        sizeof(I2O_SM_MULTIPART_MESSAGE_FRAME)))
          {
            _data.reset(new detail::ChainData(pRef));
            return;
          }

        unsigned short i2oMessageCode = pvtMsg->XFunctionCode;
        switch (i2oMessageCode)
          {

          case I2O_SM_PREAMBLE:
            {
              _data.reset(new detail::InitMsgData(pRef));
              break;
            }

          case I2O_SM_DATA:
            {
              _data.reset(new detail::EventMsgData(pRef));
              break;
            }

          case I2O_SM_DQM:
            {
              _data.reset(new detail::DQMEventMsgData(pRef));
              break;
            }

          case I2O_SM_ERROR:
            {
              _data.reset(new detail::ErrorEventMsgData(pRef));
              break;
            }

          default:
            {
              _data.reset(new detail::ChainData(pRef));
              _data->markCorrupt();
              break;
            }

          }
      }
  }

  I2OChain::I2OChain(I2OChain const& other) :
    _data(other._data)
  { }

  I2OChain::~I2OChain()
  { }

  I2OChain& I2OChain::operator=(I2OChain const& rhs)
  {
    // This is the standard copy/swap algorithm, to obtain the strong
    // exception safety guarantee.
    I2OChain temp(rhs);
    swap(temp);
    return *this;
  }
  
  void I2OChain::swap(I2OChain& other)
  {
    _data.swap(other._data);
  }

  bool I2OChain::empty() const
  {
    // We're empty if we have no ChainData, or if the ChainData object
    // we have is empty.
    return !_data || _data->empty();
  }


  bool I2OChain::complete() const
  {
    if (!_data) return false;
    return _data->complete();
  }


  bool I2OChain::faulty() const
  {
    if (!_data) return false;
    return _data->faulty();
  }


  unsigned int I2OChain::faultyBits() const
  {
    if (!_data) return 0;
    return _data->faultyBits();
  }


  void I2OChain::addToChain(I2OChain &newpart)
  {
    // fragments can not be added to empty, complete, or faulty chains.
    if (empty())
      {
        XCEPT_RAISE(stor::exception::I2OChain,
                    "A fragment may not be added to an empty chain.");
      }
    if (complete())
      {
        XCEPT_RAISE(stor::exception::I2OChain,
                    "A fragment may not be added to a complete chain.");
      }

    // empty, complete, or faulty new parts can not be added to chains
    if (newpart.empty())
      {
        XCEPT_RAISE(stor::exception::I2OChain,
                    "An empty chain may not be added to an existing chain.");
      }
    if (newpart.complete())
      {
        XCEPT_RAISE(stor::exception::I2OChain,
                    "A complete chain may not be added to an existing chain.");
      }

    // require the new part and this chain to have the same fragment key
    FragKey thisKey = fragmentKey();
    FragKey thatKey = newpart.fragmentKey();
    // should change this to != once we implement that operator in FragKey
    if (thisKey < thatKey || thatKey < thisKey)
      {
        std::stringstream msg;
        msg << "A fragment key mismatch was detected when trying to add "
            << "a chain link to an existing chain. "
            << "Existing key values = ("
            << ((int)thisKey.code_) << "," << thisKey.run_ << ","
            << thisKey.event_ << "," << thisKey.secondaryId_ << ","
            << thisKey.originatorPid_ << "," << thisKey.originatorGuid_
            << "), new key values = ("
            << ((int)thatKey.code_) << "," << thatKey.run_ << ","
            << thatKey.event_ << "," << thatKey.secondaryId_ << ","
            << thatKey.originatorPid_ << "," << thatKey.originatorGuid_
            << ").";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }

    // add the fragment to the current chain
    _data->addToChain(*(newpart._data));
    newpart.release();
  }

  //void I2OChain::markComplete()
  //{
  //  // TODO:: Should we throw an exception if _data is null? If so, what
  //  // type? Right now, we do nothing if _data is null.
  //  if (_data) _data->markComplete();
  //}

  void I2OChain::markFaulty()
  {
    // TODO:: Should we throw an exception if _data is null? If so, what
    // type? Right now, we do nothing if _data is null.
    if (_data) _data->markFaulty();
  }

  unsigned long* I2OChain::getBufferData() const
  {
    return _data ?  _data->getBufferData() : 0UL;
  }

  void I2OChain::release()
  {
    // A default-constructed chain controls no resources; we can
    // relinquish our control over any controlled Reference by
    // becoming like a default-constructed chain.
    I2OChain().swap(*this);
  }

  unsigned int I2OChain::messageCode() const
  {
    if (!_data) return Header::INVALID;
    return _data->messageCode();
  }

  unsigned int I2OChain::rbBufferId() const
  {
    if (!_data) return 0;
    return _data->rbBufferId();
  }

  unsigned int I2OChain::hltLocalId() const
  {
    if (!_data) return 0;
    return _data->hltLocalId();
  }

  unsigned int I2OChain::hltInstance() const
  {
    if (!_data) return 0;
    return _data->hltInstance();
  }

  unsigned int I2OChain::hltTid() const
  {
    if (!_data) return 0;
    return _data->hltTid();
  }

  std::string I2OChain::hltURL() const
  {
    if (!_data) return "";
    return _data->hltURL();
  }

  std::string I2OChain::hltClassName() const
  {
    if (!_data) return "";
    return _data->hltClassName();
  }

  unsigned int I2OChain::fuProcessId() const
  {
    if (!_data) return 0;
    return _data->fuProcessId();
  }

  unsigned int I2OChain::fuGuid() const
  {
    if (!_data) return 0;
    return _data->fuGuid();
  }

  FragKey I2OChain::fragmentKey() const
  {
    if (!_data) return FragKey(Header::INVALID,0,0,0,0,0);
    return _data->fragmentKey();
  }

  unsigned int I2OChain::fragmentCount() const
  {
    if (!_data) return 0;
    return _data->fragmentCount();
  }

  double I2OChain::creationTime() const
  {
    if (!_data) return -1;
    return _data->creationTime();
  }

  double I2OChain::lastFragmentTime() const
  {
    if (!_data) return -1;
    return _data->lastFragmentTime();
  }

  double I2OChain::staleWindowStartTime() const
  {
    if (!_data) return -1;
    return _data->staleWindowStartTime();
  }

  void I2OChain::addToStaleWindowStartTime(const utils::duration_t duration)
  {
    if (!_data) return;
    _data->addToStaleWindowStartTime(duration);
  }

  void I2OChain::resetStaleWindowStartTime()
  {
    if (!_data) return;
    _data->resetStaleWindowStartTime();
  }

  void I2OChain::tagForStream(StreamID streamId)
  {
    if (!_data)
      {
        std::stringstream msg;
        msg << "An empty chain can not be tagged for a specific ";
        msg << "event stream.";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    _data->tagForStream(streamId);
  }

  void I2OChain::tagForEventConsumer(QueueID queueId)
  {
    if (!_data)
      {
        std::stringstream msg;
        msg << "An empty chain can not be tagged for a specific ";
        msg << "event consumer.";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    _data->tagForEventConsumer(queueId);
  }

  void I2OChain::tagForDQMEventConsumer(QueueID queueId)
  {
    if (!_data)
      {
        std::stringstream msg;
        msg << "An empty chain can not be tagged for a specific ";
        msg << "DQM event consumer.";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    _data->tagForDQMEventConsumer(queueId);
  }

  bool I2OChain::isTaggedForAnyStream() const
  {
    if (!_data) return false;
    return _data->isTaggedForAnyStream();
  }

  bool I2OChain::isTaggedForAnyEventConsumer() const
  {
    if (!_data) return false;
    return _data->isTaggedForAnyEventConsumer();
  }

  bool I2OChain::isTaggedForAnyDQMEventConsumer() const
  {
    if (!_data) return false;
    return _data->isTaggedForAnyDQMEventConsumer();
  }

  std::vector<StreamID> I2OChain::getStreamTags() const
  {
    if (!_data)
      {
        std::vector<StreamID> tmpList;
        return tmpList;
      }
    return _data->getStreamTags();
  }

  std::vector<QueueID> I2OChain::getEventConsumerTags() const
  {
    if (!_data)
      {
        std::vector<QueueID> tmpList;
        return tmpList;
      }
    return _data->getEventConsumerTags();
  }

  std::vector<QueueID> I2OChain::getDQMEventConsumerTags() const
  {
    if (!_data)
      {
        std::vector<QueueID> tmpList;
        return tmpList;
      }
    return _data->getDQMEventConsumerTags();
  }

  unsigned long I2OChain::totalDataSize() const
  {
    if (!_data) return 0UL;
    return _data->totalDataSize();
  }

  unsigned long I2OChain::dataSize(int fragmentIndex) const
  {
    if (!_data) return 0UL;
    return _data->dataSize(fragmentIndex);
  }

  unsigned char* I2OChain::dataLocation(int fragmentIndex) const
  {
    if (!_data) return 0UL;
    return _data->dataLocation(fragmentIndex);
  }

  unsigned int I2OChain::getFragmentID(int fragmentIndex) const
  {
    if (!_data) return 0;
    return _data->getFragmentID(fragmentIndex);
  }

  unsigned long I2OChain::headerSize() const
  {
    if (!_data) return 0UL;
    return _data->headerSize();
  }

  unsigned char* I2OChain::headerLocation() const
  {
    if (!_data) return 0UL;
    return _data->headerLocation();
  }

  unsigned int I2OChain::
  copyFragmentsIntoBuffer(std::vector<unsigned char>& targetBuffer) const
  {
    if (!_data) return 0;
    return _data->copyFragmentsIntoBuffer(targetBuffer);
  }

  std::string I2OChain::outputModuleLabel() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The output module label can not be determined from an empty I2OChain.");
      }
    return _data->outputModuleLabel();
  }

  std::string I2OChain::topFolderName() const
  {
    if( !_data )
      {
        XCEPT_RAISE( stor::exception::I2OChain,
                     "The top folder name can not be determined from an empty I2OChain." );
      }
    return _data->topFolderName();
  }

  DQMKey I2OChain::dqmKey() const
  {
    if( !_data )
      {
        XCEPT_RAISE( stor::exception::I2OChain,
                     "The DQM key can not be determined from an empty I2OChain." );
      }
    return _data->dqmKey();
  }

  uint32 I2OChain::outputModuleId() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The output module ID can not be determined from an empty I2OChain.");
      }
    return _data->outputModuleId();
  }

  void I2OChain::hltTriggerNames(Strings& nameList) const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "HLT trigger names can not be determined from an empty I2OChain.");
      }
    _data->hltTriggerNames(nameList);
  }

  void I2OChain::hltTriggerSelections(Strings& nameList) const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "HLT trigger selections can not be determined from an empty I2OChain.");
      }
    _data->hltTriggerSelections(nameList);
  }

  void I2OChain::l1TriggerNames(Strings& nameList) const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "L1 trigger names can not be determined from an empty I2OChain.");
      }
    _data->l1TriggerNames(nameList);
  }

  uint32 I2OChain::hltTriggerCount() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The number of HLT trigger bits can not be determined from an empty I2OChain.");
      }
    return _data->hltTriggerCount();
  }

  void I2OChain::hltTriggerBits(std::vector<unsigned char>& bitList) const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "HLT trigger bits can not be determined from an empty I2OChain.");
      }
    _data->hltTriggerBits(bitList);
  }

  void I2OChain::assertRunNumber(uint32 runNumber)
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The run number can not be checked for an empty I2OChain.");
      }
    return _data->assertRunNumber(runNumber);
  }

  uint32 I2OChain::runNumber() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The run number can not be determined from an empty I2OChain.");
      }
    return _data->runNumber();
  }

  uint32 I2OChain::lumiSection() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The luminosity section can not be determined from an empty I2OChain.");
      }
    return _data->lumiSection();
  }

  uint32 I2OChain::eventNumber() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The event number can not be determined from an empty I2OChain.");
      }
    return _data->eventNumber();
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

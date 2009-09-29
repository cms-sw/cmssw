// $Id: $

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor
{

  namespace detail
  {

    InitMsgData::InitMsgData(toolbox::mem::Reference* pRef) :
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

  } // namespace detail

} // namespace stor

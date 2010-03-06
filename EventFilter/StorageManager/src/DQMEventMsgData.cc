// $Id: DQMEventMsgData.cc,v 1.2 2010/01/07 18:03:32 mommsen Exp $

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"

namespace stor
{

  namespace detail
  {

    DQMEventMsgData::DQMEventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(pRef, I2O_SM_DQM, Header::DQM_EVENT),
      _headerFieldsCached(false)
    {
      parseI2OHeader();
    }

    inline std::string DQMEventMsgData::do_topFolderName() const
    {

      if( faulty() || !complete() )
        {
          std::stringstream msg;
          msg << "A top folder name can not be determined from a ";
          msg << "faulty or incomplete DQM event message.";
          XCEPT_RAISE( stor::exception::IncompleteInitMessage, msg.str() );
        }

      if( !_headerFieldsCached ) {cacheHeaderFields();}
      return _topFolderName;
    }

    inline DQMKey DQMEventMsgData::do_dqmKey() const
    {

      if( faulty() || !complete() )
        {
          std::stringstream msg;
          msg << "The DQM key can not be determined from a ";
          msg << "faulty or incomplete DQM event message.";
          XCEPT_RAISE( stor::exception::IncompleteInitMessage, msg.str() );
        }

      if( !_headerFieldsCached ) {cacheHeaderFields();}
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

      boost::shared_ptr<DQMEventMsgView> msgView;
      if (_fragmentCount == 1)
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

  } // namespace detail

} // namespace stor

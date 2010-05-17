// $Id: DQMEventMsgData.cc,v 1.9 2010/05/12 12:22:06 mommsen Exp $
/// @file: DQMEventMsgData.cc

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"

namespace stor
{

  namespace detail
  {

    DQMEventMsgData::DQMEventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(I2O_SM_DQM, Header::DQM_EVENT),
      _headerFieldsCached(false)
    {
      addFirstFragment(pRef);
      parseI2OHeader();
    }

    inline size_t DQMEventMsgData::do_i2oFrameSize() const
    {
      return sizeof(I2O_SM_DQM_MESSAGE_FRAME);
    }

    std::string DQMEventMsgData::do_topFolderName() const
    {
      if( !headerOkay() )
      {
        std::stringstream msg;
        msg << "A top folder name can not be determined from a ";
        msg << "faulty or incomplete DQM event message.";
        XCEPT_RAISE( stor::exception::IncompleteDQMEventMessage, msg.str() );
      }

      if( !_headerFieldsCached ) {cacheHeaderFields();}
      return _topFolderName;
    }

    uint32_t DQMEventMsgData::do_adler32Checksum() const
    {
      if( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An adler32 checksum can not be determined from a ";
        msg << "faulty or incomplete DQM event message.";
        XCEPT_RAISE( stor::exception::IncompleteDQMEventMessage, msg.str() );
      }
      
      if( !_headerFieldsCached ) {cacheHeaderFields();}
      return _adler32;
    }

    DQMKey DQMEventMsgData::do_dqmKey() const
    {
      if( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The DQM key can not be determined from a ";
        msg << "faulty or incomplete DQM event message.";
        XCEPT_RAISE( stor::exception::IncompleteDQMEventMessage, msg.str() );
      }
      
      if( !_headerFieldsCached ) {cacheHeaderFields();}
      return _dqmKey;
    }

    uint32_t DQMEventMsgData::do_runNumber() const
    {

      if( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The run number can not be determined from a ";
        msg << "faulty or incomplete DQM event message.";
        XCEPT_RAISE( stor::exception::IncompleteDQMEventMessage, msg.str() );
      }
      
      if( !_headerFieldsCached ) {cacheHeaderFields();}
      return _dqmKey.runNumber;
    }

    uint32_t DQMEventMsgData::do_lumiSection() const
    {

      if( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The lumi section can not be determined from a ";
        msg << "faulty or incomplete DQM event message.";
        XCEPT_RAISE( stor::exception::IncompleteDQMEventMessage, msg.str() );
      }
      
      if( !_headerFieldsCached ) {cacheHeaderFields();}
      return _dqmKey.lumiSection;
    }

    void DQMEventMsgData::do_assertRunNumber(uint32_t runNumber)
    {
      if ( headerOkay() && do_runNumber() != runNumber )
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
      if ( !headerOkay() )
      {
        return 0;
      }
    
      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerSize;
    }

    unsigned char* DQMEventMsgData::do_headerLocation() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! _headerFieldsCached) {cacheHeaderFields();}
      return _headerLocation;
    }

    inline unsigned char*
    DQMEventMsgData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      if ( parsable() )
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
      if ( parsable() )
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

      _headerSize = msgView->headerSize()
        + sizeof(uint32_t); // in contrast to other message types,
                          // DQM messages do not contain the data
                          // length entry (uint32_t) in headerSize()
      _headerLocation = msgView->startAddress();
      _topFolderName = msgView->topFolderName();
      _adler32 = msgView->adler32_chksum();

      _dqmKey.runNumber = msgView->runNumber();
      _dqmKey.lumiSection = msgView->lumiSection();

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

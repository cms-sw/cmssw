// $Id: DQMEventMsgData.cc,v 1.12 2011/03/28 13:49:04 mommsen Exp $
/// @file: DQMEventMsgData.cc

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"

namespace stor
{

  namespace detail
  {

    DQMEventMsgData::DQMEventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(I2O_SM_DQM, Header::DQM_EVENT),
      headerFieldsCached_(false)
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

      if( !headerFieldsCached_ ) {cacheHeaderFields();}
      return topFolderName_;
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
      
      if( !headerFieldsCached_ ) {cacheHeaderFields();}
      return adler32_;
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
      
      if( !headerFieldsCached_ ) {cacheHeaderFields();}
      return dqmKey_;
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
      
      if( !headerFieldsCached_ ) {cacheHeaderFields();}
      return dqmKey_.runNumber;
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
      
      if( !headerFieldsCached_ ) {cacheHeaderFields();}
      return dqmKey_.lumiSection;
    }

    void DQMEventMsgData::do_assertRunNumber(uint32_t runNumber)
    {
      if ( headerOkay() && do_runNumber() != runNumber )
      {
        std::ostringstream errorMsg;
        errorMsg << "Run number " << do_runNumber() 
          << " of DQM event for LS " << do_lumiSection() <<
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
    
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerSize_;
    }

    unsigned char* DQMEventMsgData::do_headerLocation() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerLocation_;
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
          (I2O_SM_DQM_MESSAGE_FRAME*) ref_->getDataLocation();
        fragKey_.code_ = messageCode_;
        fragKey_.run_ = smMsg->runID;
        fragKey_.event_ = smMsg->eventAtUpdateID;
        fragKey_.secondaryId_ = smMsg->folderID;
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

    // Adapted from InitMsgData::cacheHeaderFields
    void DQMEventMsgData::cacheHeaderFields() const
    {

      unsigned char* firstFragLoc = dataLocation(0);

      boost::shared_ptr<DQMEventMsgView> msgView;
      if (fragmentCount_ == 1)
      {
        msgView.reset(new DQMEventMsgView(firstFragLoc));
      }
      else
      {
        copyFragmentsIntoBuffer(headerCopy_);
        msgView.reset(new DQMEventMsgView(&headerCopy_[0]));
      }

      headerSize_ = msgView->headerSize()
        + sizeof(uint32_t); // in contrast to other message types,
                          // DQM messages do not contain the data
                          // length entry (uint32_t) in headerSize()
      headerLocation_ = msgView->startAddress();
      topFolderName_ = msgView->topFolderName();
      adler32_ = msgView->adler32_chksum();

      dqmKey_.runNumber = msgView->runNumber();
      dqmKey_.lumiSection = msgView->lumiSection();
      dqmKey_.topLevelFolderName = msgView->topFolderName();

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

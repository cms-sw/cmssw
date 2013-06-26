// $Id: ErrorEventData.cc,v 1.7 2011/03/07 15:31:32 mommsen Exp $
/// @file: ErrorEventData.cc

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/FRDEventMessage.h"

namespace stor
{

  namespace detail
  {

    ErrorEventMsgData::ErrorEventMsgData(toolbox::mem::Reference* pRef) :
      ChainData(I2O_SM_ERROR, Header::ERROR_EVENT),
      headerFieldsCached_(false)
    {
      addFirstFragment(pRef);
      parseI2OHeader();
    }

    inline size_t ErrorEventMsgData::do_i2oFrameSize() const
    {
      return sizeof(I2O_SM_DATA_MESSAGE_FRAME);
    }

    unsigned long ErrorEventMsgData::do_headerSize() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerSize_;
    }

    unsigned char* ErrorEventMsgData::do_headerLocation() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }

      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerLocation_;
    }

    inline unsigned char*
    ErrorEventMsgData::do_fragmentLocation(unsigned char* dataLoc) const
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

    void
    ErrorEventMsgData::do_assertRunNumber(uint32_t runNumber)
    {
      if ( headerOkay() && do_runNumber() != runNumber )
      {
        std::ostringstream errorMsg;
        errorMsg << "Run number " << do_runNumber() 
          << " of error event " << do_eventNumber() <<
          " received from " << hltURL() << 
          " (FU process id " << fuProcessId() << ")" <<
          " does not match the run number " << runNumber << 
          " used to configure the StorageManager." <<
          " Enforce usage of configured run number.";
        runNumber_ = runNumber;
        XCEPT_RAISE(stor::exception::RunNumberMismatch, errorMsg.str());
      }
    }

    uint32_t ErrorEventMsgData::do_runNumber() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "A run number can not be determined from a ";
        msg << "faulty or incomplete ErrorEvent message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return runNumber_;
    }

    uint32_t ErrorEventMsgData::do_lumiSection() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "A luminosity section can not be determined from a ";
        msg << "faulty or incomplete ErrorEvent message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return lumiSection_;
    }

    uint32_t ErrorEventMsgData::do_eventNumber() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An event number can not be determined from a ";
        msg << "faulty or incomplete ErrorEvent message.";
        XCEPT_RAISE(stor::exception::IncompleteEventMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return eventNumber_;
    }

    inline void ErrorEventMsgData::parseI2OHeader()
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

    void ErrorEventMsgData::cacheHeaderFields() const
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
      // the full Event message header  (FRD events have fixed header
      // size, so the check is easy)
      else if (firstFragSize > sizeof(FRDEventHeader_V2))
      {
        useFirstFrag = true;
      }

      boost::shared_ptr<FRDEventMsgView> msgView;
      if (useFirstFrag)
      {
        msgView.reset(new FRDEventMsgView(firstFragLoc));
      }
      else
      {
        copyFragmentsIntoBuffer(headerCopy_);
        msgView.reset(new FRDEventMsgView(&headerCopy_[0]));
      }

      headerSize_ = sizeof(FRDEventHeader_V2);
      headerLocation_ = msgView->startAddress();

      runNumber_ = msgView->run();
      lumiSection_ = msgView->lumi();
      eventNumber_ = msgView->event();

      headerFieldsCached_ = true;
    }

  } // namespace detail

} // namespace stor


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

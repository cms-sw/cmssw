// $Id: InitMsgData.cc,v 1.9 2012/04/20 10:48:02 mommsen Exp $
/// @file: InitMsgData.cc

#include "EventFilter/StorageManager/src/ChainData.h"

#include "IOPool/Streamer/interface/InitMessage.h"

namespace stor
{

  namespace detail
  {

    InitMsgData::InitMsgData(toolbox::mem::Reference* pRef) :
      ChainData(I2O_SM_PREAMBLE, Header::INIT),
      headerFieldsCached_(false)
    {
      addFirstFragment(pRef);
      parseI2OHeader();
    }

    inline size_t InitMsgData::do_i2oFrameSize() const
    {
      return sizeof(I2O_SM_PREAMBLE_MESSAGE_FRAME);
    }

    unsigned long InitMsgData::do_headerSize() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerSize_;
    }

    unsigned char* InitMsgData::do_headerLocation() const
    {
      if ( !headerOkay() )
      {
        return 0;
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return headerLocation_;
    }

    inline unsigned char*
    InitMsgData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      if ( parsable() )
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

    uint32_t InitMsgData::do_adler32Checksum() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An adler32 checksum can not be determined from a ";
        msg << "faulty or incomplete INIT message.";
        XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return adler32_;
    }

    std::string InitMsgData::do_outputModuleLabel() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An output module label can not be determined from a ";
        msg << "faulty or incomplete INIT message.";
        XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return outputModuleLabel_;
    }

    uint32_t InitMsgData::do_outputModuleId() const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "An output module ID can not be determined from a ";
        msg << "faulty or incomplete INIT message.";
        XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      return outputModuleId_;
    }

    void InitMsgData::do_hltTriggerNames(Strings& nameList) const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The HLT trigger names can not be determined from a ";
        msg << "faulty or incomplete INIT message.";
        XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      nameList = hltTriggerNames_;
    }

    void InitMsgData::do_hltTriggerSelections(Strings& nameList) const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The HLT trigger selections can not be determined from a ";
        msg << "faulty or incomplete INIT message.";
        XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      nameList = hltTriggerSelections_;
    }

    void InitMsgData::do_l1TriggerNames(Strings& nameList) const
    {
      if ( !headerOkay() )
      {
        std::stringstream msg;
        msg << "The L1 trigger names can not be determined from a ";
        msg << "faulty or incomplete INIT message.";
        XCEPT_RAISE(stor::exception::IncompleteInitMessage, msg.str());
      }
      
      if (! headerFieldsCached_) {cacheHeaderFields();}
      nameList = l1TriggerNames_;
    }

    inline void InitMsgData::parseI2OHeader()
    {
      if ( parsable() )
      {
        I2O_SM_PREAMBLE_MESSAGE_FRAME *smMsg =
          (I2O_SM_PREAMBLE_MESSAGE_FRAME*) ref_->getDataLocation();
        fragKey_.code_ = messageCode_;
        fragKey_.run_ = 0;
        fragKey_.event_ = smMsg->hltTid;
        fragKey_.secondaryId_ = smMsg->outModID;
        fragKey_.originatorPid_ = smMsg->fuProcID;
        fragKey_.originatorGuid_ = smMsg->fuGUID;
        rbBufferId_ = smMsg->rbBufferID;
        hltLocalId_ = smMsg->hltLocalId;
        hltInstance_ = smMsg->hltInstance;
        hltTid_ = smMsg->hltTid;
        fuProcessId_ = smMsg->fuProcID;
        fuGuid_ = smMsg->fuGUID;
        nExpectedEPs_ = smMsg->nExpectedEPs;
      }
    }

    void InitMsgData::cacheHeaderFields() const
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
        copyFragmentsIntoBuffer(headerCopy_);
        msgView.reset(new InitMsgView(&headerCopy_[0]));
      }
      
      headerSize_ = msgView->headerSize();
      headerLocation_ = msgView->startAddress();
      adler32_ = msgView->adler32_chksum();
      outputModuleLabel_ = msgView->outputModuleLabel();
      outputModuleId_ = msgView->outputModuleId();
      msgView->hltTriggerNames(hltTriggerNames_);
      msgView->hltTriggerSelections(hltTriggerSelections_);
      msgView->l1TriggerNames(l1TriggerNames_);

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

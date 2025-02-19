// $Id: I2OChain.cc,v 1.27 2012/04/20 10:48:02 mommsen Exp $
/// @file: I2OChain.cc

#include <algorithm>

#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

#include "EventFilter/StorageManager/src/ChainData.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "IOPool/Streamer/interface/MsgHeader.h"

#include "interface/evb/i2oEVBMsgs.h"
#include "interface/shared/i2oXFunctionCodes.h"
#include "interface/shared/version.h"


namespace stor
{

  // A default-constructed I2OChain has a null (shared) pointer.
  I2OChain::I2OChain():
    data_()
  {}

  I2OChain::I2OChain(toolbox::mem::Reference* pRef)
  {
    if (pRef)
      {
        I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
          (I2O_PRIVATE_MESSAGE_FRAME*) pRef->getDataLocation();
        if (!pvtMsg)
          {
            data_.reset(new detail::ChainData());
            data_->addFirstFragment(pRef);
            return;
          }

        unsigned short i2oMessageCode = pvtMsg->XFunctionCode;
        switch (i2oMessageCode)
          {

          case I2O_SM_PREAMBLE:
            {
              data_.reset(new detail::InitMsgData(pRef));
              break;
            }

          case I2O_SM_DATA:
            {
              data_.reset(new detail::EventMsgData(pRef));
              break;
            }

          case I2O_SM_DQM:
            {
              data_.reset(new detail::DQMEventMsgData(pRef));
              break;
            }

          case I2O_EVM_LUMISECTION:
            {
              data_.reset(new detail::EndLumiSectMsgData(pRef));
              break;
            }

          case I2O_SM_ERROR:
            {
              data_.reset(new detail::ErrorEventMsgData(pRef));
              break;
            }

          default:
            {
              data_.reset(new detail::ChainData());
              data_->addFirstFragment(pRef);
              data_->markCorrupt();
              break;
            }

          }
      }
  }

  I2OChain::I2OChain(I2OChain const& other) :
    data_(other.data_)
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
    data_.swap(other.data_);
  }

  bool I2OChain::empty() const
  {
    // We're empty if we have no ChainData, or if the ChainData object
    // we have is empty.
    return !data_ || data_->empty();
  }


  bool I2OChain::complete() const
  {
    if (!data_) return false;
    return data_->complete();
  }


  bool I2OChain::faulty() const
  {
    if (!data_) return false;
    return data_->faulty();
  }


  unsigned int I2OChain::faultyBits() const
  {
    if (!data_) return 0;
    return data_->faultyBits();
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
    data_->addToChain(*(newpart.data_));
    newpart.release();
  }

  //void I2OChain::markComplete()
  //{
  //  // TODO:: Should we throw an exception if data_ is null? If so, what
  //  // type? Right now, we do nothing if data_ is null.
  //  if (data_) data_->markComplete();
  //}

  void I2OChain::markFaulty()
  {
    // TODO:: Should we throw an exception if data_ is null? If so, what
    // type? Right now, we do nothing if data_ is null.
    if (data_) data_->markFaulty();
  }

  unsigned long* I2OChain::getBufferData() const
  {
    return data_ ?  data_->getBufferData() : 0UL;
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
    if (!data_) return Header::INVALID;
    return data_->messageCode();
  }

  unsigned short I2OChain::i2oMessageCode() const
  {
    if (!data_) return 0xffff;
    return data_->i2oMessageCode();
  }

  unsigned int I2OChain::rbBufferId() const
  {
    if (!data_) return 0;
    return data_->rbBufferId();
  }

  unsigned int I2OChain::hltLocalId() const
  {
    if (!data_) return 0;
    return data_->hltLocalId();
  }

  unsigned int I2OChain::hltInstance() const
  {
    if (!data_) return 0;
    return data_->hltInstance();
  }

  unsigned int I2OChain::hltTid() const
  {
    if (!data_) return 0;
    return data_->hltTid();
  }

  std::string I2OChain::hltURL() const
  {
    if (!data_) return "";
    return data_->hltURL();
  }

  std::string I2OChain::hltClassName() const
  {
    if (!data_) return "";
    return data_->hltClassName();
  }

  unsigned int I2OChain::fuProcessId() const
  {
    if (!data_) return 0;
    return data_->fuProcessId();
  }

  unsigned int I2OChain::fuGuid() const
  {
    if (!data_) return 0;
    return data_->fuGuid();
  }

  FragKey I2OChain::fragmentKey() const
  {
    if (!data_) return FragKey(Header::INVALID,0,0,0,0,0);
    return data_->fragmentKey();
  }

  unsigned int I2OChain::fragmentCount() const
  {
    if (!data_) return 0;
    return data_->fragmentCount();
  }

  utils::TimePoint_t I2OChain::creationTime() const
  {
    if (!data_) return boost::posix_time::not_a_date_time;
    return data_->creationTime();
  }

  utils::TimePoint_t I2OChain::lastFragmentTime() const
  {
    if (!data_) return boost::posix_time::not_a_date_time;
    return data_->lastFragmentTime();
  }

  utils::TimePoint_t I2OChain::staleWindowStartTime() const
  {
    if (!data_) return boost::posix_time::not_a_date_time;
    return data_->staleWindowStartTime();
  }

  void I2OChain::addToStaleWindowStartTime(const utils::Duration_t duration)
  {
    if (!data_) return;
    data_->addToStaleWindowStartTime(duration);
  }

  void I2OChain::resetStaleWindowStartTime()
  {
    if (!data_) return;
    data_->resetStaleWindowStartTime();
  }

  void I2OChain::tagForStream(StreamID streamId)
  {
    if (!data_)
      {
        std::stringstream msg;
        msg << "An empty chain can not be tagged for a specific ";
        msg << "event stream.";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    data_->tagForStream(streamId);
  }

  void I2OChain::tagForEventConsumer(QueueID queueId)
  {
    if (!data_)
      {
        std::stringstream msg;
        msg << "An empty chain can not be tagged for a specific ";
        msg << "event consumer.";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    data_->tagForEventConsumer(queueId);
  }

  void I2OChain::tagForDQMEventConsumer(QueueID queueId)
  {
    if (!data_)
      {
        std::stringstream msg;
        msg << "An empty chain can not be tagged for a specific ";
        msg << "DQM event consumer.";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    data_->tagForDQMEventConsumer(queueId);
  }

  bool I2OChain::isTaggedForAnyStream() const
  {
    if (!data_) return false;
    return data_->isTaggedForAnyStream();
  }

  bool I2OChain::isTaggedForAnyEventConsumer() const
  {
    if (!data_) return false;
    return data_->isTaggedForAnyEventConsumer();
  }

  bool I2OChain::isTaggedForAnyDQMEventConsumer() const
  {
    if (!data_) return false;
    return data_->isTaggedForAnyDQMEventConsumer();
  }

  std::vector<StreamID> I2OChain::getStreamTags() const
  {
    if (!data_)
      {
        std::vector<StreamID> tmpList;
        return tmpList;
      }
    return data_->getStreamTags();
  }

  QueueIDs I2OChain::getEventConsumerTags() const
  {
    if (!data_)
      {
        QueueIDs tmpList;
        return tmpList;
      }
    return data_->getEventConsumerTags();
  }

  QueueIDs I2OChain::getDQMEventConsumerTags() const
  {
    if (!data_)
      {
        QueueIDs tmpList;
        return tmpList;
      }
    return data_->getDQMEventConsumerTags();
  }

  unsigned int I2OChain::droppedEventsCount() const
  {
    if (!data_)
      {
        std::stringstream msg;
        msg << "A dropped event count cannot be retrieved from an empty chain";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    return data_->droppedEventsCount();
  }

  void I2OChain::setDroppedEventsCount(unsigned int count)
  {
    if (!data_)
      {
        std::stringstream msg;
        msg << "A dropped event count cannot be added to an empty chain";
        XCEPT_RAISE(stor::exception::I2OChain, msg.str());
      }
    data_->setDroppedEventsCount(count);
  }

  size_t I2OChain::memoryUsed() const
  {
    if (!data_) return 0;
    return data_->memoryUsed();
  }

  unsigned long I2OChain::totalDataSize() const
  {
    if (!data_) return 0UL;
    return data_->totalDataSize();
  }

  unsigned long I2OChain::dataSize(int fragmentIndex) const
  {
    if (!data_) return 0UL;
    return data_->dataSize(fragmentIndex);
  }

  unsigned char* I2OChain::dataLocation(int fragmentIndex) const
  {
    if (!data_) return 0UL;
    return data_->dataLocation(fragmentIndex);
  }

  unsigned int I2OChain::getFragmentID(int fragmentIndex) const
  {
    if (!data_) return 0;
    return data_->getFragmentID(fragmentIndex);
  }

  unsigned long I2OChain::headerSize() const
  {
    if (!data_) return 0UL;
    return data_->headerSize();
  }

  unsigned char* I2OChain::headerLocation() const
  {
    if (!data_) return 0UL;
    return data_->headerLocation();
  }

  unsigned int I2OChain::
  copyFragmentsIntoBuffer(std::vector<unsigned char>& targetBuffer) const
  {
    if (!data_) return 0;
    return data_->copyFragmentsIntoBuffer(targetBuffer);
  }

  std::string I2OChain::topFolderName() const
  {
    if( !data_ )
      {
        XCEPT_RAISE( stor::exception::I2OChain,
                     "The top folder name can not be determined from an empty I2OChain." );
      }
    return data_->topFolderName();
  }

  DQMKey I2OChain::dqmKey() const
  {
    if( !data_ )
      {
        XCEPT_RAISE( stor::exception::I2OChain,
                     "The DQM key can not be determined from an empty I2OChain." );
      }
    return data_->dqmKey();
  }

  std::string I2OChain::outputModuleLabel() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The output module label can not be determined from an empty I2OChain.");
      }
    return data_->outputModuleLabel();
  }

  uint32_t I2OChain::outputModuleId() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The output module ID can not be determined from an empty I2OChain.");
      }
    return data_->outputModuleId();
  }

  uint32_t I2OChain::nExpectedEPs() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The slave EP count can not be determined from an empty I2OChain.");
      }
    return data_->nExpectedEPs();
  }

  void I2OChain::hltTriggerNames(Strings& nameList) const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "HLT trigger names can not be determined from an empty I2OChain.");
      }
    data_->hltTriggerNames(nameList);
  }

  void I2OChain::hltTriggerSelections(Strings& nameList) const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "HLT trigger selections can not be determined from an empty I2OChain.");
      }
    data_->hltTriggerSelections(nameList);
  }

  void I2OChain::l1TriggerNames(Strings& nameList) const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "L1 trigger names can not be determined from an empty I2OChain.");
      }
    data_->l1TriggerNames(nameList);
  }

  uint32_t I2OChain::hltTriggerCount() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The number of HLT trigger bits can not be determined from an empty I2OChain.");
      }
    return data_->hltTriggerCount();
  }

  void I2OChain::hltTriggerBits(std::vector<unsigned char>& bitList) const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "HLT trigger bits can not be determined from an empty I2OChain.");
      }
    data_->hltTriggerBits(bitList);
  }

  void I2OChain::assertRunNumber(uint32_t runNumber)
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The run number can not be checked for an empty I2OChain.");
      }
    return data_->assertRunNumber(runNumber);
  }

  uint32_t I2OChain::runNumber() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The run number can not be determined from an empty I2OChain.");
      }
    return data_->runNumber();
  }

  uint32_t I2OChain::lumiSection() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The luminosity section can not be determined from an empty I2OChain.");
      }
    return data_->lumiSection();
  }

  uint32_t I2OChain::eventNumber() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The event number can not be determined from an empty I2OChain.");
      }
    return data_->eventNumber();
  }

  uint32_t I2OChain::adler32Checksum() const
  {
    if (!data_)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The adler32 checksum can not be determined from an empty I2OChain.");
      }
    return data_->adler32Checksum();
  }

  bool I2OChain::isEndOfLumiSectionMessage() const
  {
    if (!data_) return false;
    return data_->isEndOfLumiSectionMessage();
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

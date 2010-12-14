// $Id: I2OChain.cc,v 1.21 2010/05/17 15:59:10 mommsen Exp $
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
    _data()
  {}

  I2OChain::I2OChain(toolbox::mem::Reference* pRef)
  {
    if (pRef)
      {
        I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
          (I2O_PRIVATE_MESSAGE_FRAME*) pRef->getDataLocation();
        if (!pvtMsg)
          {
            _data.reset(new detail::ChainData());
            _data->addFirstFragment(pRef);
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

          case I2O_EVM_LUMISECTION:
            {
              _data.reset(new detail::EndLumiSectMsgData(pRef));
              break;
            }

          case I2O_SM_ERROR:
            {
              _data.reset(new detail::ErrorEventMsgData(pRef));
              break;
            }

          default:
            {
              _data.reset(new detail::ChainData());
              _data->addFirstFragment(pRef);
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

  unsigned short I2OChain::i2oMessageCode() const
  {
    if (!_data) return 0xffff;
    return _data->i2oMessageCode();
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

  utils::time_point_t I2OChain::creationTime() const
  {
    if (!_data) return boost::posix_time::not_a_date_time;
    return _data->creationTime();
  }

  utils::time_point_t I2OChain::lastFragmentTime() const
  {
    if (!_data) return boost::posix_time::not_a_date_time;
    return _data->lastFragmentTime();
  }

  utils::time_point_t I2OChain::staleWindowStartTime() const
  {
    if (!_data) return boost::posix_time::not_a_date_time;
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

  size_t I2OChain::memoryUsed() const
  {
    if (!_data) return 0;
    return _data->memoryUsed();
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

  uint32_t I2OChain::outputModuleId() const
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

  uint32_t I2OChain::hltTriggerCount() const
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

  void I2OChain::assertRunNumber(uint32_t runNumber)
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The run number can not be checked for an empty I2OChain.");
      }
    return _data->assertRunNumber(runNumber);
  }

  uint32_t I2OChain::runNumber() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The run number can not be determined from an empty I2OChain.");
      }
    return _data->runNumber();
  }

  uint32_t I2OChain::lumiSection() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The luminosity section can not be determined from an empty I2OChain.");
      }
    return _data->lumiSection();
  }

  uint32_t I2OChain::eventNumber() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The event number can not be determined from an empty I2OChain.");
      }
    return _data->eventNumber();
  }

  uint32_t I2OChain::adler32Checksum() const
  {
    if (!_data)
      {
        XCEPT_RAISE(stor::exception::I2OChain,
          "The adler32 checksum can not be determined from an empty I2OChain.");
      }
    return _data->adler32Checksum();
  }

  bool I2OChain::isEndOfLumiSectionMessage() const
  {
    if (!_data) return false;
    return _data->isEndOfLumiSectionMessage();
  }

} // namespace stor

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

// $Id: I2OChain.cc,v 1.4 2009/06/29 15:46:09 mommsen Exp $

#include <algorithm>
#include "EventFilter/StorageManager/interface/Exception.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"
#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

namespace stor
{
  namespace detail
  {
    ///////////////////////////////////////////////////////////////////
    //
    // class ChainData is responsible for managing a single chain of
    // References and associated status information (such as tags
    // applied to the 'event' that lives in the buffer(s) managed by the
    // Reference(s).
    //
    // Only one ChainData object ever manages a given
    // Reference. Furthermore, ChainData makes use of References such
    // that a duplicate of a given Reference is never made. Thus when a
    // ChainData object is destroyed, any and all References managed by
    // that object are immediately released.
    //
    ///////////////////////////////////////////////////////////////////
    class ChainData
    {
      enum BitMasksForFaulty { INVALID_INITIAL_REFERENCE = 0x1,
                               CORRUPT_INITIAL_HEADER = 0x2,
                               INVALID_SECONDARY_REFERENCE = 0x4,
                               CORRUPT_SECONDARY_HEADER = 0x8,
                               TOTAL_COUNT_MISMATCH = 0x10,
                               FRAGMENTS_OUT_OF_ORDER = 0x20,
                               DUPLICATE_FRAGMENT = 0x40,
                               INCOMPLETE_MESSAGE = 0x80,
                               EXTERNALLY_REQUESTED = 0x10000 };


    public:
      explicit ChainData(toolbox::mem::Reference* pRef);
      virtual ~ChainData();
      bool empty() const;
      bool complete() const;
      bool faulty() const;
      unsigned int faultyBits() const;
      bool parsable() const;
      void addToChain(ChainData const& newpart);
      void markComplete();
      void markFaulty();
      void markCorrupt();
      unsigned long* getBufferData() const;
      void swap(ChainData& other);
      unsigned int messageCode() const {return _messageCode;}
      FragKey const& fragmentKey() const {return _fragKey;}
      unsigned int fragmentCount() const {return _fragmentCount;}
      unsigned int rbBufferId() const {return _rbBufferId;}
      unsigned int hltLocalId() const {return _hltLocalId;}
      unsigned int hltInstance() const {return _hltInstance;}
      unsigned int hltTid() const {return _hltTid;}
      unsigned int fuProcessId() const {return _fuProcessId;}
      unsigned int fuGuid() const {return _fuGuid;}
      utils::time_point_t creationTime() const {return _creationTime;}
      utils::time_point_t lastFragmentTime() const {return _lastFragmentTime;}
      utils::time_point_t staleWindowStartTime() const {return _staleWindowStartTime;}
      void addToStaleWindowStartTime(const utils::duration_t duration) {
        _staleWindowStartTime += duration;
      }
      void resetStaleWindowStartTime() {
        _staleWindowStartTime = utils::getCurrentTime();
      }
      unsigned long totalDataSize() const;
      unsigned long dataSize(int fragmentIndex) const;
      unsigned char* dataLocation(int fragmentIndex) const;
      unsigned int getFragmentID(int fragmentIndex) const;
      unsigned int copyFragmentsIntoBuffer(std::vector<unsigned char>& buff) const;

      unsigned long headerSize() const;
      unsigned char* headerLocation() const;

      std::string hltURL() const;
      std::string hltClassName() const;
      uint32 outputModuleId() const;

      std::string outputModuleLabel() const;
      void hltTriggerNames(Strings& nameList) const;
      void hltTriggerSelections(Strings& nameList) const;
      void l1TriggerNames(Strings& nameList) const;

      void assertRunNumber(uint32 runNumber);

      uint32 runNumber() const;
      uint32 lumiSection() const;
      uint32 eventNumber() const;

      std::string topFolderName() const;
      DQMKey dqmKey() const;

      uint32 hltTriggerCount() const;
      void hltTriggerBits(std::vector<unsigned char>& bitList) const;

      void tagForStream(StreamID);
      void tagForEventConsumer(QueueID);
      void tagForDQMEventConsumer(QueueID);
      bool isTaggedForAnyStream() const {return !_streamTags.empty();}
      bool isTaggedForAnyEventConsumer() const {return !_eventConsumerTags.empty();}
      bool isTaggedForAnyDQMEventConsumer() const {return !_dqmEventConsumerTags.empty();}
      std::vector<StreamID> const& getStreamTags() const;
      std::vector<QueueID> const& getEventConsumerTags() const;
      std::vector<QueueID> const& getDQMEventConsumerTags() const;

    private:
      std::vector<StreamID> _streamTags;
      std::vector<QueueID> _eventConsumerTags;
      std::vector<QueueID> _dqmEventConsumerTags;

    protected:
      toolbox::mem::Reference* _ref;

      bool _complete;
      unsigned int _faultyBits;

      unsigned int _messageCode;
      FragKey _fragKey;
      unsigned int _fragmentCount;
      unsigned int _expectedNumberOfFragments;
      unsigned int _rbBufferId;
      unsigned int _hltLocalId;
      unsigned int _hltInstance;
      unsigned int _hltTid;
      unsigned int _fuProcessId;
      unsigned int _fuGuid;

      utils::time_point_t _creationTime;
      utils::time_point_t _lastFragmentTime;
      utils::time_point_t _staleWindowStartTime;

      bool validateDataLocation(toolbox::mem::Reference* ref,
                                BitMasksForFaulty maskToUse);
      bool validateMessageSize(toolbox::mem::Reference* ref,
                               BitMasksForFaulty maskToUse);
      bool validateFragmentIndexAndCount(toolbox::mem::Reference* ref,
                                         BitMasksForFaulty maskToUse);
      bool validateExpectedFragmentCount(toolbox::mem::Reference* ref,
                                         BitMasksForFaulty maskToUse);
      bool validateFragmentOrder(toolbox::mem::Reference* ref,
                                 int& indexValue);
      bool validateMessageCode(toolbox::mem::Reference* ref,
                               unsigned short expectedI2OMessageCode);

      virtual unsigned long do_headerSize() const;
      virtual unsigned char* do_headerLocation() const;

      virtual unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      virtual uint32 do_outputModuleId() const;

      virtual std::string do_outputModuleLabel() const;
      virtual void do_hltTriggerNames(Strings& nameList) const;
      virtual void do_hltTriggerSelections(Strings& nameList) const;
      virtual void do_l1TriggerNames(Strings& nameList) const;

      virtual std::string do_topFolderName() const;
      virtual DQMKey do_dqmKey() const;

      virtual uint32 do_hltTriggerCount() const;
      virtual void do_hltTriggerBits(std::vector<unsigned char>& bitList) const;

      virtual void do_assertRunNumber(uint32 runNumber);

      virtual uint32 do_runNumber() const;
      virtual uint32 do_lumiSection() const;
      virtual uint32 do_eventNumber() const;
    };

    // A ChainData object may or may not contain a Reference.
    inline ChainData::ChainData(toolbox::mem::Reference* pRef) :
      _streamTags(),
      _eventConsumerTags(),
      _dqmEventConsumerTags(),
      _ref(pRef),
      _complete(false),
      _faultyBits(0),
      _messageCode(Header::INVALID),
      _fragKey(Header::INVALID,0,0,0,0,0),
      _fragmentCount(0),
      _expectedNumberOfFragments(0),
      _rbBufferId(0),
      _hltLocalId(0),
      _hltInstance(0),
      _hltTid(0),
      _fuProcessId(0),
      _fuGuid(0),
      _creationTime(-1),
      _lastFragmentTime(-1),
      _staleWindowStartTime(-1)
    {
      // Avoid the situation in which all unparsable chains
      // have the same fragment key.  We do this by providing a 
      // variable default value for one of the fragKey fields.
      if (pRef)
        {
          _fragKey.secondaryId_ = (uint32) pRef->getDataLocation();
        }
      else
        {
          _fragKey.secondaryId_ = (uint32) time(0);
        }

      if (pRef)
        {
          _creationTime = utils::getCurrentTime();
          _lastFragmentTime = _creationTime;
          _staleWindowStartTime = _creationTime;

          // first fragment in Reference chain
          ++_fragmentCount;
          int workingIndex = -1;

          if (validateDataLocation(pRef, INVALID_INITIAL_REFERENCE) &&
              validateMessageSize(pRef, CORRUPT_INITIAL_HEADER) &&
              validateFragmentIndexAndCount(pRef, CORRUPT_INITIAL_HEADER))
            {
              I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
                (I2O_SM_MULTIPART_MESSAGE_FRAME*) pRef->getDataLocation();
              _expectedNumberOfFragments = smMsg->numFrames;

              validateFragmentOrder(pRef, workingIndex);
            }

          // subsequent fragments in Reference chain
          toolbox::mem::Reference* curRef = pRef->getNextReference();
          while (curRef)
            {
              ++_fragmentCount;

              if (validateDataLocation(curRef, INVALID_SECONDARY_REFERENCE) &&
                  validateMessageSize(curRef, CORRUPT_SECONDARY_HEADER) &&
                  validateFragmentIndexAndCount(curRef, CORRUPT_SECONDARY_HEADER))
                {
                  validateExpectedFragmentCount(curRef, TOTAL_COUNT_MISMATCH);
                  validateFragmentOrder(curRef, workingIndex);
                }

              curRef = curRef->getNextReference();
            }
        }
    }

    // A ChainData that has a Reference is in charge of releasing
    // it. Because releasing a Reference can throw an exception, we have
    // to be prepared to swallow the exception. This is fairly gross,
    // because we lose any exception information. But allowing an
    // exception to escape from a destructor is even worse, so we must
    // do what we must do.
    //
    inline ChainData::~ChainData()
    {
      if (_ref) 
        {
          //std::cout << std::endl << std::endl << std::hex
          //          << "### releasing 0x" << ((int) _ref)
          //          << std::dec << std::endl << std::endl;
          try { _ref->release(); }
          catch (...) { /* swallow any exception. */ }
        }
    }

    inline bool ChainData::empty() const
    {
      return !_ref;
    }

    inline bool ChainData::complete() const
    {
      return _complete;
    }

    inline bool ChainData::faulty() const
    {
      return (_faultyBits != 0);
    }

    inline unsigned int ChainData::faultyBits() const
    {
      return _faultyBits;
    }

    inline bool ChainData::parsable() const
    {
      return (_ref) && ((_faultyBits & INVALID_INITIAL_REFERENCE) == 0) &&
        ((_faultyBits & CORRUPT_INITIAL_HEADER) == 0);
    }

    // this method currently does NOT support operation on empty chains
    // (both the existing chain and the new part must be non-empty!)
    void ChainData::addToChain(ChainData const& newpart)
    {
      // if either the current chain or the newpart are faulty, we
      // simply append the new stuff to the end of the existing chain
      if (faulty() || newpart.faulty())
        {
          // update our fragment count to include the new fragments
          toolbox::mem::Reference* curRef = newpart._ref;
          while (curRef) {
            ++_fragmentCount;
            curRef = curRef->getNextReference();
          }

          // append the new fragments to the end of the existing chain
          toolbox::mem::Reference* lastRef = _ref;
          curRef = _ref->getNextReference();
          while (curRef) {
            lastRef = curRef;
            curRef = curRef->getNextReference();
          }
          lastRef->setNextReference(newpart._ref->duplicate());

          // merge the faulty flags from the new part into the existing flags
          _faultyBits |= newpart._faultyBits;

          // update the time stamps
          _lastFragmentTime = utils::getCurrentTime();
          _staleWindowStartTime = _lastFragmentTime;
          if (newpart.creationTime() < _creationTime)
          {
            _creationTime = newpart.creationTime();
          }

          return;
        }

      // loop over the fragments in the new part
      toolbox::mem::Reference* newRef = newpart._ref;
      while (newRef)
        {
          // unlink the next element in the new chain from that chain
          toolbox::mem::Reference* nextNewRef = newRef->getNextReference();
          newRef->setNextReference(0);

          // if the new fragment that we're working with is the first one
          // in its chain, we need to duplicate it (now that it is unlinked)
          // This is necessary since it is still being managed by an I2OChain
          // somewhere.  The subsequent fragments in the new part do not
          // need to be duplicated since we explicitly unlinked them from
          // the first one.
          if (newRef == newpart._ref) {newRef = newpart._ref->duplicate();}

          // we want to track whether the fragment was added (it *always* should be)
          bool fragmentWasAdded = false;

          // determine the index of the new fragment
          I2O_SM_MULTIPART_MESSAGE_FRAME *thatMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) newRef->getDataLocation();
          unsigned int newIndex = thatMsg->frameCount;
          //std::cout << "newIndex = " << newIndex << std::endl;

          // verify that the total fragment counts match
          unsigned int newFragmentTotalCount = thatMsg->numFrames;
          if (newFragmentTotalCount != _expectedNumberOfFragments)
            {
              _faultyBits |= TOTAL_COUNT_MISMATCH;
            }

          // if the new fragment goes at the head of the chain, handle that here
          I2O_SM_MULTIPART_MESSAGE_FRAME *fragMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) _ref->getDataLocation();
          unsigned int firstIndex = fragMsg->frameCount;
          //std::cout << "firstIndex = " << firstIndex << std::endl;
          if (newIndex < firstIndex)
            {
              newRef->setNextReference(_ref);
              _ref = newRef;
              fragmentWasAdded = true;
            }

          else
            {
              // loop over the existing fragments and insert the new one
              // in the correct place
              toolbox::mem::Reference* curRef = _ref;
              for (unsigned int idx = 0; idx < _fragmentCount; ++idx)
                {
                  // if we have a duplicate fragment, add it after the existing
                  // one and indicate the error
                  I2O_SM_MULTIPART_MESSAGE_FRAME *curMsg =
                    (I2O_SM_MULTIPART_MESSAGE_FRAME*) curRef->getDataLocation();
                  unsigned int curIndex = curMsg->frameCount;
                  //std::cout << "curIndex = " << curIndex << std::endl;
                  if (newIndex == curIndex) 
                    {
                      _faultyBits |= DUPLICATE_FRAGMENT;
                      newRef->setNextReference(curRef->getNextReference());
                      curRef->setNextReference(newRef);
                      fragmentWasAdded = true;
                      break;
                    }

                  // if we have reached the end of the chain, add the
                  // new fragment to the end
                  //std::cout << "nextRef = " << ((int) nextRef) << std::endl;
                  toolbox::mem::Reference* nextRef = curRef->getNextReference();
                  if (nextRef == 0)
                    {
                      curRef->setNextReference(newRef);
                      fragmentWasAdded = true;
                      break;
                    }

                  I2O_SM_MULTIPART_MESSAGE_FRAME *nextMsg =
                    (I2O_SM_MULTIPART_MESSAGE_FRAME*) nextRef->getDataLocation();
                  unsigned int nextIndex = nextMsg->frameCount;
                  //std::cout << "nextIndex = " << nextIndex << std::endl;
                  if (newIndex > curIndex && newIndex < nextIndex)
                    {
                      newRef->setNextReference(curRef->getNextReference());
                      curRef->setNextReference(newRef);
                      fragmentWasAdded = true;
                      break;
                    }

                  curRef = nextRef;
                }
            }

          // update the fragment count and check if the chain is now complete
          if (!fragmentWasAdded)
            {
              // this should never happen - if it does, there is a logic
              // error in the loop above
              XCEPT_RAISE(stor::exception::I2OChain,
                          "A fragment was unable to be added to a chain.");
            }
          ++_fragmentCount;
          
          newRef = nextNewRef;
        }

      // update the time stamps
      _lastFragmentTime = utils::getCurrentTime();
      _staleWindowStartTime = _lastFragmentTime;
      if (newpart.creationTime() < _creationTime)
      {
        _creationTime = newpart.creationTime();
      }
      
      if (!faulty() && _fragmentCount == _expectedNumberOfFragments)
        {
          markComplete();
        }
    }

    inline void ChainData::markComplete()
    {
      _complete = true;
    }

    inline void ChainData::markFaulty()
    {
      _faultyBits |= EXTERNALLY_REQUESTED;
    }

    inline void ChainData::markCorrupt()
    {
      _faultyBits |= CORRUPT_INITIAL_HEADER;
    }

    inline unsigned long* ChainData::getBufferData() const
    {
      return _ref 
        ?  static_cast<unsigned long*>(_ref->getDataLocation()) 
        : 0UL;
    }

    inline void ChainData::swap(ChainData& other)
    {
      _streamTags.swap(other._streamTags);
      _eventConsumerTags.swap(other._eventConsumerTags);
      _dqmEventConsumerTags.swap(other._dqmEventConsumerTags);
      std::swap(_ref, other._ref);
      std::swap(_complete, other._complete);
      std::swap(_faultyBits, other._faultyBits);
      std::swap(_messageCode, other._messageCode);
      std::swap(_fragKey, other._fragKey);
      std::swap(_fragmentCount, other._fragmentCount);
      std::swap(_expectedNumberOfFragments, other._expectedNumberOfFragments);
      std::swap(_creationTime, other._creationTime);
      std::swap(_lastFragmentTime, other._lastFragmentTime);
      std::swap(_staleWindowStartTime, other._staleWindowStartTime);
    }

    unsigned long ChainData::totalDataSize() const
    {
      unsigned long totalSize = 0;
      toolbox::mem::Reference* curRef = _ref;
      for (unsigned int idx = 0; idx < _fragmentCount; ++idx)
        {
          I2O_MESSAGE_FRAME *i2oMsg =
            (I2O_MESSAGE_FRAME*) curRef->getDataLocation();
          if (!faulty())
            {
              I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
                (I2O_SM_MULTIPART_MESSAGE_FRAME*) i2oMsg;
              totalSize += smMsg->dataSize;
            }
          else if (i2oMsg)
            {
              totalSize += (i2oMsg->MessageSize*4);
            }

          curRef = curRef->getNextReference();
        }
      return totalSize;
    }

    unsigned long ChainData::dataSize(int fragmentIndex) const
    {
      toolbox::mem::Reference* curRef = _ref;
      for (int idx = 0; idx < fragmentIndex; ++idx)
        {
          curRef = curRef->getNextReference();
        }

      I2O_MESSAGE_FRAME *i2oMsg =
        (I2O_MESSAGE_FRAME*) curRef->getDataLocation();
      if (!faulty())
        {
          I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) i2oMsg;
          return smMsg->dataSize;
        }
      else if (i2oMsg)
        {
          return (i2oMsg->MessageSize*4);
        }
      return 0;
    }

    unsigned char* ChainData::dataLocation(int fragmentIndex) const
    {
      toolbox::mem::Reference* curRef = _ref;
      for (int idx = 0; idx < fragmentIndex; ++idx)
        {
          curRef = curRef->getNextReference();
        }

      if (!faulty())
        {
          return do_fragmentLocation(static_cast<unsigned char*>
                                     (curRef->getDataLocation()));
        }
      else
        {
          return static_cast<unsigned char*>(curRef->getDataLocation());
        }
    }

    unsigned int ChainData::getFragmentID(int fragmentIndex) const
    {
      toolbox::mem::Reference* curRef = _ref;
      for (int idx = 0; idx < fragmentIndex; ++idx)
        {
          curRef = curRef->getNextReference();
        }

      if (parsable())
        {
          I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) curRef->getDataLocation();
          return smMsg->frameCount;
        }
      else
        {
          return 0;
        }
    }

    unsigned int ChainData::
    copyFragmentsIntoBuffer(std::vector<unsigned char>& targetBuffer) const
    {
      unsigned long fullSize = totalDataSize();
      if (targetBuffer.capacity() < fullSize)
        {
          targetBuffer.resize(fullSize);
        }
      unsigned char* targetLoc = (unsigned char*)&targetBuffer[0];

      toolbox::mem::Reference* curRef = _ref;
      while (curRef)
        {
          unsigned char* fragmentLoc =
            (unsigned char*) curRef->getDataLocation();
          unsigned long sourceSize = 0;
          unsigned char* sourceLoc = 0;

          if (!faulty())
            {
              I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
                (I2O_SM_MULTIPART_MESSAGE_FRAME*) fragmentLoc;
              sourceSize = smMsg->dataSize;
              sourceLoc = do_fragmentLocation(fragmentLoc);
            }
          else if (fragmentLoc)
            {
              I2O_MESSAGE_FRAME *i2oMsg = (I2O_MESSAGE_FRAME*) fragmentLoc;
              sourceSize = i2oMsg->MessageSize * 4;
              sourceLoc = fragmentLoc;
            }

          if (sourceSize > 0)
            {
              std::copy(sourceLoc, sourceLoc+sourceSize, targetLoc);
              targetLoc += sourceSize;
            }

          curRef = curRef->getNextReference();
        }

      return static_cast<unsigned int>(fullSize);
    }

    inline unsigned long ChainData::headerSize() const
    {
      return do_headerSize();
    }

    inline unsigned char* ChainData::headerLocation() const
    {
      return do_headerLocation();
    }

    inline std::string ChainData::hltURL() const
    {
      if (parsable())
        {
          I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) _ref->getDataLocation();
          size_t size = std::min(strlen(smMsg->hltURL),
                                 (size_t) MAX_I2O_SM_URLCHARS);
          std::string URL(smMsg->hltURL, size);
          return URL;
        }
      else
        {
          return "";
        }
    }

    inline std::string ChainData::hltClassName() const
    {
      if (parsable())
        {
          I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) _ref->getDataLocation();
          size_t size = std::min(strlen(smMsg->hltClassName),
                                 (size_t) MAX_I2O_SM_URLCHARS);
          std::string className(smMsg->hltClassName, size);
          return className;
        }
      else
        {
          return "";
        }
    }

    inline uint32 ChainData::outputModuleId() const
    {
      return do_outputModuleId();
    }


    inline std::string ChainData::outputModuleLabel() const
    {
      return do_outputModuleLabel();
    }

    inline std::string ChainData::topFolderName() const
    {
      return do_topFolderName();
    }

    inline DQMKey ChainData::dqmKey() const
    {
      return do_dqmKey();
    }

    inline void ChainData::hltTriggerNames(Strings& nameList) const
    {
      do_hltTriggerNames(nameList);
    }

    inline void ChainData::hltTriggerSelections(Strings& nameList) const
    {
      do_hltTriggerSelections(nameList);
    }

    inline void ChainData::l1TriggerNames(Strings& nameList) const
    {
      do_l1TriggerNames(nameList);
    }

    inline uint32 ChainData::hltTriggerCount() const
    {
      return do_hltTriggerCount();
    }

    inline void
    ChainData::hltTriggerBits(std::vector<unsigned char>& bitList) const
    {
      do_hltTriggerBits(bitList);
    }

    inline void ChainData::assertRunNumber(uint32 runNumber)
    {
      do_assertRunNumber(runNumber);
    }

    inline uint32 ChainData::runNumber() const
    {
      return do_runNumber();
    }

    inline uint32 ChainData::lumiSection() const
    {
      return do_lumiSection();
    }

    inline uint32 ChainData::eventNumber() const
    {
      return do_eventNumber();
    }

    inline void ChainData::tagForStream(StreamID streamId)
    {
      _streamTags.push_back(streamId);
    }

    inline void ChainData::tagForEventConsumer(QueueID queueId)
    {
      _eventConsumerTags.push_back(queueId);
    }

    inline void ChainData::tagForDQMEventConsumer(QueueID queueId)
    {
      _dqmEventConsumerTags.push_back(queueId);
    }

    inline std::vector<StreamID> const& ChainData::getStreamTags() const
    {
      return _streamTags;
    }

    inline std::vector<QueueID> const& ChainData::getEventConsumerTags() const
    {
      return _eventConsumerTags;
    }

    inline std::vector<QueueID> const& ChainData::getDQMEventConsumerTags() const
    {
      return _dqmEventConsumerTags;
    }


    inline bool
    ChainData::validateDataLocation(toolbox::mem::Reference* ref,
                                    BitMasksForFaulty maskToUse)
    {
      I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
        (I2O_PRIVATE_MESSAGE_FRAME*) ref->getDataLocation();
      if (!pvtMsg)
        {
          _faultyBits |= maskToUse;
          return false;
        }
      return true;
    }

    inline bool
    ChainData::validateMessageSize(toolbox::mem::Reference* ref,
                                   BitMasksForFaulty maskToUse)
    {
      I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
        (I2O_PRIVATE_MESSAGE_FRAME*) ref->getDataLocation();
      if ((size_t)(pvtMsg->StdMessageFrame.MessageSize*4) <
          sizeof(I2O_SM_MULTIPART_MESSAGE_FRAME))
        {
          _faultyBits |= maskToUse;
          return false;
        }
      return true;
    }

    inline bool
    ChainData::validateFragmentIndexAndCount(toolbox::mem::Reference* ref,
                                             BitMasksForFaulty maskToUse)
    {
      I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
        (I2O_SM_MULTIPART_MESSAGE_FRAME*) ref->getDataLocation();
      if (smMsg->numFrames < 1 || smMsg->frameCount >= smMsg->numFrames)
        {
          _faultyBits |= maskToUse;
          return false;
        }
      return true;
    }

    inline bool
    ChainData::validateExpectedFragmentCount(toolbox::mem::Reference* ref,
                                             BitMasksForFaulty maskToUse)
    {
      I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
        (I2O_SM_MULTIPART_MESSAGE_FRAME*) ref->getDataLocation();
      if (smMsg->numFrames != _expectedNumberOfFragments)
        {
          _faultyBits |= maskToUse;
          return false;
        }
      return true;
    }

    inline bool
    ChainData::validateFragmentOrder(toolbox::mem::Reference* ref,
                                     int& indexValue)
    {
      int problemCount = 0;
      I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
        (I2O_SM_MULTIPART_MESSAGE_FRAME*) ref->getDataLocation();
      int thisIndex = static_cast<int>(smMsg->frameCount);
      if (thisIndex == indexValue)
        {
          _faultyBits |= DUPLICATE_FRAGMENT;
          ++problemCount;
        }
      else if (thisIndex < indexValue)
        {
          _faultyBits |= FRAGMENTS_OUT_OF_ORDER;
          ++problemCount;
        }
      indexValue = thisIndex;
      return (problemCount == 0);
    }

    inline bool
    ChainData::validateMessageCode(toolbox::mem::Reference* ref,
                                   unsigned short expectedI2OMessageCode)
    {
      I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
        (I2O_PRIVATE_MESSAGE_FRAME*) ref->getDataLocation();
      if (pvtMsg->XFunctionCode != expectedI2OMessageCode)
        {
          _faultyBits |= CORRUPT_SECONDARY_HEADER;
          return false;
        }
      return true;
    }

    inline unsigned long ChainData::do_headerSize() const
    {
      return 0;
    }

    inline unsigned char* ChainData::do_headerLocation() const
    {
      return 0;
    }

    inline unsigned char*
    ChainData::do_fragmentLocation(unsigned char* dataLoc) const
    {
      return dataLoc;
    }

    inline uint32 ChainData::do_outputModuleId() const
    {
      std::stringstream msg;
      msg << "An output module ID is only available from a valid, ";
      msg << "complete INIT or Event message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline std::string ChainData::do_outputModuleLabel() const
    {
      std::stringstream msg;
      msg << "An output module label is only available from a valid, ";
      msg << "complete INIT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline std::string ChainData::do_topFolderName() const
    {
      std::stringstream msg;
      msg << "A top folder name is only available from a valid, ";
      msg << "complete DQM event message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline DQMKey ChainData::do_dqmKey() const
    {
      std::stringstream msg;
      msg << "The DQM key is only available from a valid, ";
      msg << "complete DQM event message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline void ChainData::do_hltTriggerNames(Strings& nameList) const
    {
      std::stringstream msg;
      msg << "The HLT trigger names are only available from a valid, ";
      msg << "complete INIT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline void ChainData::do_hltTriggerSelections(Strings& nameList) const
    {
      std::stringstream msg;
      msg << "The HLT trigger selections are only available from a valid, ";
      msg << "complete INIT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline void ChainData::do_l1TriggerNames(Strings& nameList) const
    {
      std::stringstream msg;
      msg << "The L1 trigger names are only available from a valid, ";
      msg << "complete INIT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline uint32 ChainData::do_hltTriggerCount() const
    {
      std::stringstream msg;
      msg << "An HLT trigger count is only available from a valid, ";
      msg << "complete Event message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline void 
    ChainData::do_hltTriggerBits(std::vector<unsigned char>& bitList) const
    {
      std::stringstream msg;
      msg << "The HLT trigger bits are only available from a valid, ";
      msg << "complete Event message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline void ChainData::do_assertRunNumber(uint32 runNumber)
    {}

    inline uint32 ChainData::do_runNumber() const
    {
      std::stringstream msg;
      msg << "A run number is only available from a valid, ";
      msg << "complete EVENT or ERROR_EVENT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline uint32 ChainData::do_lumiSection() const
    {
      std::stringstream msg;
      msg << "A luminosity section is only available from a valid, ";
      msg << "complete EVENT or ERROR_EVENT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }

    inline uint32 ChainData::do_eventNumber() const
    {
      std::stringstream msg;
      msg << "An event number is only available from a valid, ";
      msg << "complete EVENT or ERROR_EVENT message.";
      XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
    }


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
        _runNumber = runNumber;
        std::ostringstream errorMsg;
        errorMsg << "Run number " << do_runNumber() 
          << " of error event " << do_eventNumber() <<
          " received from " << hltURL() << 
          " (FU process id " << fuProcessId() << ")" <<
          " does not match the run number " << runNumber << 
          " used to configure the StorageManager." <<
          " Enforce usage of configured run number.";
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

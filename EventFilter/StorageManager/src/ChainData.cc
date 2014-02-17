// $Id: ChainData.cc,v 1.20 2012/04/20 10:48:01 mommsen Exp $
/// @file: ChainData.cc

#include "IOPool/Streamer/interface/MsgHeader.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/StreamID.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "EventFilter/StorageManager/src/ChainData.h"

#include "interface/shared/i2oXFunctionCodes.h"
#include "interface/shared/version.h"

#include <stdlib.h>
#include "zlib.h"


using namespace stor;

// A ChainData object may or may not contain a Reference.
detail::ChainData::ChainData(const unsigned short i2oMessageCode,
                             const unsigned int messageCode) :
  streamTags_(),
  eventConsumerTags_(),
  dqmEventConsumerTags_(),
  ref_(0),
  complete_(false),
  faultyBits_(INCOMPLETE_MESSAGE),
  messageCode_(messageCode),
  i2oMessageCode_(i2oMessageCode),
  fragKey_(Header::INVALID,0,0,0,0,0),
  fragmentCount_(0),
  expectedNumberOfFragments_(0),
  rbBufferId_(0),
  hltLocalId_(0),
  hltInstance_(0),
  hltTid_(0),
  fuProcessId_(0),
  fuGuid_(0)
{
  #ifdef STOR_DEBUG_CORRUPT_MESSAGES
  double r = rand()/static_cast<double>(RAND_MAX);
  if (r < 0.001)
  {
    // std::cout << "Simulating corrupt I2O message" << std::endl;
    // markCorrupt();
  }
  else if (r < 0.02)
  {
    std::cout << "Simulating faulty I2O message" << std::endl;
    markFaulty();
  }
  #endif // STOR_DEBUG_CORRUPT_MESSAGES
}

// A ChainData that has a Reference is in charge of releasing
// it. Because releasing a Reference can throw an exception, we have
// to be prepared to swallow the exception. This is fairly gross,
// because we lose any exception information. But allowing an
// exception to escape from a destructor is even worse, so we must
// do what we must do.
//
detail::ChainData::~ChainData()
{
  if (ref_) 
    {
      //std::cout << std::endl << std::endl << std::hex
      //          << "### releasing 0x" << ((int) ref_)
      //          << std::dec << std::endl << std::endl;
      try { ref_->release(); }
      catch (...) { /* swallow any exception. */ }
    }
}

bool detail::ChainData::empty() const
{
  return !ref_;
}

bool detail::ChainData::complete() const
{
  return complete_;
}

bool detail::ChainData::faulty() const
{
  if (complete_)
    return (faultyBits_ != 0);
  else
    return (faultyBits_ != INCOMPLETE_MESSAGE);
}

unsigned int detail::ChainData::faultyBits() const
{
  return faultyBits_;
}

bool detail::ChainData::parsable() const
{
  return (ref_) && 
    ((faultyBits_ & INVALID_INITIAL_REFERENCE & ~INCOMPLETE_MESSAGE) == 0) &&
    ((faultyBits_ & CORRUPT_INITIAL_HEADER & ~INCOMPLETE_MESSAGE) == 0);
}

bool detail::ChainData::headerOkay() const
{
  return ( (faultyBits_ & ~WRONG_CHECKSUM) == 0);
}

void detail::ChainData::addFirstFragment(toolbox::mem::Reference* pRef)
{
  if (ref_)
  {
    XCEPT_RAISE(stor::exception::I2OChain, "Cannot add a first fragment to a non-empty I2OChain.");
  }
  ref_ = pRef;

  // Avoid the situation in which all unparsable chains
  // have the same fragment key.  We do this by providing a 
  // variable default value for one of the fragKey fields.
  if (pRef)
    {
      fragKey_.secondaryId_ = static_cast<uint32_t>(
        (uintptr_t)pRef->getDataLocation()
      );
    }
  else
    {
      fragKey_.secondaryId_ = static_cast<uint32_t>( time(0) );
    }

  if (pRef)
    {
      creationTime_ = utils::getCurrentTime();
      lastFragmentTime_ = creationTime_;
      staleWindowStartTime_ = creationTime_;

      // first fragment in Reference chain
      ++fragmentCount_;
      int workingIndex = -1;

      if (validateDataLocation(pRef, INVALID_INITIAL_REFERENCE) &&
	  validateMessageSize(pRef, CORRUPT_INITIAL_HEADER) &&
	  validateFragmentIndexAndCount(pRef, CORRUPT_INITIAL_HEADER))
	{
	  I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
	    (I2O_SM_MULTIPART_MESSAGE_FRAME*) pRef->getDataLocation();
	  expectedNumberOfFragments_ = smMsg->numFrames;
	  validateFragmentOrder(pRef, workingIndex);
	}

      // subsequent fragments in Reference chain
      toolbox::mem::Reference* curRef = pRef->getNextReference();
      while (curRef)
	{
	  ++fragmentCount_;
	  
	  if (validateDataLocation(curRef, INVALID_SECONDARY_REFERENCE) &&
	      validateMessageSize(curRef, CORRUPT_SECONDARY_HEADER) &&
	      validateFragmentIndexAndCount(curRef, CORRUPT_SECONDARY_HEADER))
	    {
	      validateExpectedFragmentCount(curRef, TOTAL_COUNT_MISMATCH);
	      validateFragmentOrder(curRef, workingIndex);
              validateMessageCode(curRef, i2oMessageCode_);
 	    }
	  
	  curRef = curRef->getNextReference();
	}
    }

  checkForCompleteness();
}

void detail::ChainData::addToChain(ChainData const& newpart)
{
  if ( this->empty() )
  {
    addFirstFragment(newpart.ref_);
    return;
  }

  if (parsable() && newpart.parsable())
  {
    // loop over the fragments in the new part
    toolbox::mem::Reference* newRef = newpart.ref_;
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
      if (newRef == newpart.ref_) {newRef = newpart.ref_->duplicate();}
      
      // we want to track whether the fragment was added (it *always* should be)
      bool fragmentWasAdded = false;
      
      // determine the index of the new fragment
      I2O_SM_MULTIPART_MESSAGE_FRAME *thatMsg =
	(I2O_SM_MULTIPART_MESSAGE_FRAME*) newRef->getDataLocation();
      unsigned int newIndex = thatMsg->frameCount;
      //std::cout << "newIndex = " << newIndex << std::endl;
      
      // verify that the total fragment counts match
      unsigned int newFragmentTotalCount = thatMsg->numFrames;
      if (newFragmentTotalCount != expectedNumberOfFragments_)
      {
        faultyBits_ |= TOTAL_COUNT_MISMATCH;
      }
      
      // if the new fragment goes at the head of the chain, handle that here
      I2O_SM_MULTIPART_MESSAGE_FRAME *fragMsg =
	(I2O_SM_MULTIPART_MESSAGE_FRAME*) ref_->getDataLocation();
      unsigned int firstIndex = fragMsg->frameCount;
      //std::cout << "firstIndex = " << firstIndex << std::endl;
      if (newIndex < firstIndex)
      {
        newRef->setNextReference(ref_);
        ref_ = newRef;
        fragmentWasAdded = true;
      }

      else
      {
        // loop over the existing fragments and insert the new one
        // in the correct place
        toolbox::mem::Reference* curRef = ref_;
        for (unsigned int idx = 0; idx < fragmentCount_; ++idx)
        {
          // if we have a duplicate fragment, add it after the existing
          // one and indicate the error
          I2O_SM_MULTIPART_MESSAGE_FRAME *curMsg =
            (I2O_SM_MULTIPART_MESSAGE_FRAME*) curRef->getDataLocation();
          unsigned int curIndex = curMsg->frameCount;
          //std::cout << "curIndex = " << curIndex << std::endl;
          if (newIndex == curIndex) 
          {
            faultyBits_ |= DUPLICATE_FRAGMENT;
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
      ++fragmentCount_;
      
      newRef = nextNewRef;
    }
  }
  else
  {
    // if either the current chain or the newpart are nor parsable,
    // we simply append the new stuff to the end of the existing chain
    
    // update our fragment count to include the new fragments
    toolbox::mem::Reference* curRef = newpart.ref_;
    while (curRef) {
      ++fragmentCount_;
      curRef = curRef->getNextReference();
    }
    
    // append the new fragments to the end of the existing chain
    toolbox::mem::Reference* lastRef = ref_;
    curRef = ref_->getNextReference();
    while (curRef) {
      lastRef = curRef;
      curRef = curRef->getNextReference();
    }
    lastRef->setNextReference(newpart.ref_->duplicate());
    
    // update the time stamps
    lastFragmentTime_ = utils::getCurrentTime();
    staleWindowStartTime_ = lastFragmentTime_;
    if (newpart.creationTime() < creationTime_)
    {
      creationTime_ = newpart.creationTime();
    }
    
    return;
  }

  // merge the faulty flags from the new part into the existing flags
  faultyBits_ |= newpart.faultyBits_;

  // update the time stamps
  lastFragmentTime_ = utils::getCurrentTime();
  staleWindowStartTime_ = lastFragmentTime_;
  if (newpart.creationTime() < creationTime_)
  {
    creationTime_ = newpart.creationTime();
  }
  
  checkForCompleteness();
}

void detail::ChainData::checkForCompleteness()
{
  if ((fragmentCount_ == expectedNumberOfFragments_) &&
    ((faultyBits_ & (TOTAL_COUNT_MISMATCH | FRAGMENTS_OUT_OF_ORDER | DUPLICATE_FRAGMENT)) == 0))
    markComplete();
}

void detail::ChainData::markComplete()
{
  faultyBits_ &= ~INCOMPLETE_MESSAGE; // reset incomplete bit
  complete_ = true;
  validateAdler32Checksum();
}

void detail::ChainData::markFaulty()
{
  faultyBits_ |= EXTERNALLY_REQUESTED;
}

void detail::ChainData::markCorrupt()
{
  faultyBits_ |= CORRUPT_INITIAL_HEADER;
}

unsigned long* detail::ChainData::getBufferData() const
{
  return ref_ 
    ?  static_cast<unsigned long*>(ref_->getDataLocation()) 
    : 0UL;
}

void detail::ChainData::swap(ChainData& other)
{
  streamTags_.swap(other.streamTags_);
  eventConsumerTags_.swap(other.eventConsumerTags_);
  dqmEventConsumerTags_.swap(other.dqmEventConsumerTags_);
  std::swap(ref_, other.ref_);
  std::swap(complete_, other.complete_);
  std::swap(faultyBits_, other.faultyBits_);
  std::swap(messageCode_, other.messageCode_);
  std::swap(i2oMessageCode_, other.i2oMessageCode_);
  std::swap(fragKey_, other.fragKey_);
  std::swap(fragmentCount_, other.fragmentCount_);
  std::swap(expectedNumberOfFragments_, other.expectedNumberOfFragments_);
  std::swap(creationTime_, other.creationTime_);
  std::swap(lastFragmentTime_, other.lastFragmentTime_);
  std::swap(staleWindowStartTime_, other.staleWindowStartTime_);
}

size_t detail::ChainData::memoryUsed() const
{
  size_t memoryUsed = 0;
  toolbox::mem::Reference* curRef = ref_;
  while (curRef)
    {
      memoryUsed += curRef->getDataSize();
      curRef = curRef->getNextReference();
    }
  return memoryUsed;
}

unsigned long detail::ChainData::totalDataSize() const
{
  unsigned long totalSize = 0;
  toolbox::mem::Reference* curRef = ref_;
  for (unsigned int idx = 0; idx < fragmentCount_; ++idx)
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

unsigned long detail::ChainData::dataSize(int fragmentIndex) const
{
  toolbox::mem::Reference* curRef = ref_;
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

unsigned char* detail::ChainData::dataLocation(int fragmentIndex) const
{
  toolbox::mem::Reference* curRef = ref_;
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

unsigned int detail::ChainData::getFragmentID(int fragmentIndex) const
{
  toolbox::mem::Reference* curRef = ref_;
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

unsigned int detail::ChainData::
copyFragmentsIntoBuffer(std::vector<unsigned char>& targetBuffer) const
{
  unsigned long fullSize = totalDataSize();
  if (targetBuffer.capacity() < fullSize)
    {
      targetBuffer.resize(fullSize);
    }
  unsigned char* targetLoc = (unsigned char*)&targetBuffer[0];

  toolbox::mem::Reference* curRef = ref_;
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

unsigned long detail::ChainData::headerSize() const
{
  return do_headerSize();
}

unsigned char* detail::ChainData::headerLocation() const
{
  return do_headerLocation();
}

std::string detail::ChainData::hltURL() const
{
  if (parsable())
    {
      I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
	(I2O_SM_MULTIPART_MESSAGE_FRAME*) ref_->getDataLocation();
      size_t size = std::min(strlen(smMsg->hltURL),
			     (size_t) MAX_I2O_SM_URLCHARS);
      std::string URL(smMsg->hltURL, size);
      return URL;
    }
  else
    {
      return "unavailable";
    }
}

std::string detail::ChainData::hltClassName() const
{
  if (parsable())
    {
      I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
	(I2O_SM_MULTIPART_MESSAGE_FRAME*) ref_->getDataLocation();
      size_t size = std::min(strlen(smMsg->hltClassName),
			     (size_t) MAX_I2O_SM_URLCHARS);
      std::string className(smMsg->hltClassName, size);
      return className;
    }
  else
    {
      return "unavailable";
    }
}

std::string detail::ChainData::outputModuleLabel() const
{
  return do_outputModuleLabel();
}

uint32_t detail::ChainData::outputModuleId() const
{
  return do_outputModuleId();
}

uint32_t detail::ChainData::nExpectedEPs() const
{
  return do_nExpectedEPs();
}

std::string detail::ChainData::topFolderName() const
{
  return do_topFolderName();
}

DQMKey detail::ChainData::dqmKey() const
{
  return do_dqmKey();
}

void detail::ChainData::hltTriggerNames(Strings& nameList) const
{
  do_hltTriggerNames(nameList);
}

void detail::ChainData::hltTriggerSelections(Strings& nameList) const
{
  do_hltTriggerSelections(nameList);
}

void detail::ChainData::l1TriggerNames(Strings& nameList) const
{
  do_l1TriggerNames(nameList);
}

uint32_t detail::ChainData::hltTriggerCount() const
{
  return do_hltTriggerCount();
}

void
detail::ChainData::hltTriggerBits(std::vector<unsigned char>& bitList) const
{
  do_hltTriggerBits(bitList);
}

void detail::ChainData::assertRunNumber(uint32_t runNumber)
{
  do_assertRunNumber(runNumber);
}

uint32_t detail::ChainData::runNumber() const
{
  return do_runNumber();
}

uint32_t detail::ChainData::lumiSection() const
{
  return do_lumiSection();
}

uint32_t detail::ChainData::eventNumber() const
{
  return do_eventNumber();
}

uint32_t detail::ChainData::adler32Checksum() const
{
  return do_adler32Checksum();
}

void detail::ChainData::tagForStream(StreamID streamId)
{
  streamTags_.push_back(streamId);
}

void detail::ChainData::tagForEventConsumer(QueueID queueId)
{
  eventConsumerTags_.push_back(queueId);
}

void detail::ChainData::tagForDQMEventConsumer(QueueID queueId)
{
  dqmEventConsumerTags_.push_back(queueId);
}

std::vector<StreamID> const& detail::ChainData::getStreamTags() const
{
  return streamTags_;
}

QueueIDs const& detail::ChainData::getEventConsumerTags() const
{
  return eventConsumerTags_;
}

QueueIDs const& detail::ChainData::getDQMEventConsumerTags() const
{
  return dqmEventConsumerTags_;
}

unsigned int detail::ChainData::droppedEventsCount() const
{
  return do_droppedEventsCount();
}

void detail::ChainData::setDroppedEventsCount(unsigned int count)
{
  do_setDroppedEventsCount(count);
}

bool detail::ChainData::isEndOfLumiSectionMessage() const
{
  return ( i2oMessageCode_ == I2O_EVM_LUMISECTION );
}

bool
detail::ChainData::validateDataLocation(toolbox::mem::Reference* ref,
					BitMasksForFaulty maskToUse)
{
  I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
    (I2O_PRIVATE_MESSAGE_FRAME*) ref->getDataLocation();
  if (!pvtMsg)
    {
      faultyBits_ |= maskToUse;
      return false;
    }
  return true;
}

bool
detail::ChainData::validateMessageSize(toolbox::mem::Reference* ref,
				       BitMasksForFaulty maskToUse)
{
  I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
    (I2O_PRIVATE_MESSAGE_FRAME*) ref->getDataLocation();
  if ((size_t)(pvtMsg->StdMessageFrame.MessageSize*4) <
      sizeof(I2O_SM_MULTIPART_MESSAGE_FRAME))
    {
      faultyBits_ |= maskToUse;
      return false;
    }
  return true;
}

bool
detail::ChainData::validateFragmentIndexAndCount(toolbox::mem::Reference* ref,
						 BitMasksForFaulty maskToUse)
{
  I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
    (I2O_SM_MULTIPART_MESSAGE_FRAME*) ref->getDataLocation();
  if (smMsg->numFrames < 1 || smMsg->frameCount >= smMsg->numFrames)
    {
      faultyBits_ |= maskToUse;
      return false;
    }
  return true;
}

bool
detail::ChainData::validateExpectedFragmentCount(toolbox::mem::Reference* ref,
						 BitMasksForFaulty maskToUse)
{
  I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
    (I2O_SM_MULTIPART_MESSAGE_FRAME*) ref->getDataLocation();
  if (smMsg->numFrames != expectedNumberOfFragments_)
    {
      faultyBits_ |= maskToUse;
      return false;
    }
  return true;
}

bool
detail::ChainData::validateFragmentOrder(toolbox::mem::Reference* ref,
					 int& indexValue)
{
  int problemCount = 0;
  I2O_SM_MULTIPART_MESSAGE_FRAME *smMsg =
    (I2O_SM_MULTIPART_MESSAGE_FRAME*) ref->getDataLocation();
  int thisIndex = static_cast<int>(smMsg->frameCount);
  if (thisIndex == indexValue)
    {
      faultyBits_ |= DUPLICATE_FRAGMENT;
      ++problemCount;
    }
  else if (thisIndex < indexValue)
    {
      faultyBits_ |= FRAGMENTS_OUT_OF_ORDER;
      ++problemCount;
    }
  indexValue = thisIndex;
  return (problemCount == 0);
}

bool
detail::ChainData::validateMessageCode(toolbox::mem::Reference* ref,
				       unsigned short expectedI2OMessageCode)
{
  I2O_PRIVATE_MESSAGE_FRAME *pvtMsg =
    (I2O_PRIVATE_MESSAGE_FRAME*) ref->getDataLocation();
  if (pvtMsg->XFunctionCode != expectedI2OMessageCode)
    {
      faultyBits_ |= CORRUPT_SECONDARY_HEADER;
      return false;
    }
  return true;
}

bool detail::ChainData::validateAdler32Checksum()
{
  if ( !complete() || !headerOkay() ) return false;

  const uint32_t expected = adler32Checksum();
  if (expected == 0) return false; // Adler32 not available

  const uint32_t calculated = calculateAdler32();

  if ( calculated != expected )
  {
    faultyBits_ |= WRONG_CHECKSUM;
    return false;
  }
  return true;
}

uint32_t detail::ChainData::calculateAdler32() const
{
  uint32_t adler = adler32(0L, 0, 0);
  
  toolbox::mem::Reference* curRef = ref_;

  I2O_SM_MULTIPART_MESSAGE_FRAME* smMsg =
    (I2O_SM_MULTIPART_MESSAGE_FRAME*) curRef->getDataLocation();

  //skip event header in first fragment
  const unsigned long headerSize = do_headerSize();
  unsigned long offset = 0;

  const unsigned long payloadSize = curRef->getDataSize() - do_i2oFrameSize();
  if ( headerSize > payloadSize )
  {
    // Header continues into next fragment
    offset = headerSize - payloadSize;
  }
  else
  {
    const unsigned char* dataLocation = 
      do_fragmentLocation((unsigned char*)curRef->getDataLocation()) + headerSize;
    adler = adler32(adler, dataLocation, smMsg->dataSize - headerSize);
  }

  curRef = curRef->getNextReference();

  while (curRef)
  {
    smMsg = (I2O_SM_MULTIPART_MESSAGE_FRAME*) curRef->getDataLocation();

    const unsigned long payloadSize = curRef->getDataSize() - do_i2oFrameSize();
    if ( offset > payloadSize )
    {
      offset -= payloadSize;
    }
    else
    {
      const unsigned char* dataLocation =
        do_fragmentLocation((unsigned char*)curRef->getDataLocation()) + offset;
      adler = adler32(adler, dataLocation, smMsg->dataSize - offset);
      offset = 0;
    }

    curRef = curRef->getNextReference();
  }

  return adler;
}

std::string detail::ChainData::do_outputModuleLabel() const
{
  std::stringstream msg;
  msg << "An output module label is only available from a valid, ";
  msg << "complete INIT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32_t detail::ChainData::do_outputModuleId() const
{
  std::stringstream msg;
  msg << "An output module ID is only available from a valid, ";
  msg << "complete INIT or Event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32_t detail::ChainData::do_nExpectedEPs() const
{
  std::stringstream msg;
  msg << "The number of slave EPs is only available from a valid, ";
  msg << "complete INIT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

std::string detail::ChainData::do_topFolderName() const
{
  std::stringstream msg;
  msg << "A top folder name is only available from a valid, ";
  msg << "complete DQM event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

DQMKey detail::ChainData::do_dqmKey() const
{
  std::stringstream msg;
  msg << "The DQM key is only available from a valid, ";
  msg << "complete DQM event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

void detail::ChainData::do_hltTriggerNames(Strings& nameList) const
{
  std::stringstream msg;
  msg << "The HLT trigger names are only available from a valid, ";
  msg << "complete INIT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

void detail::ChainData::do_hltTriggerSelections(Strings& nameList) const
{
  std::stringstream msg;
  msg << "The HLT trigger selections are only available from a valid, ";
  msg << "complete INIT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

void detail::ChainData::do_l1TriggerNames(Strings& nameList) const
{
  std::stringstream msg;
  msg << "The L1 trigger names are only available from a valid, ";
  msg << "complete INIT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32_t detail::ChainData::do_hltTriggerCount() const
{
  std::stringstream msg;
  msg << "An HLT trigger count is only available from a valid, ";
  msg << "complete Event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

void 
detail::ChainData::do_hltTriggerBits(std::vector<unsigned char>& bitList) const
{
  std::stringstream msg;
  msg << "The HLT trigger bits are only available from a valid, ";
  msg << "complete Event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

unsigned int
detail::ChainData::do_droppedEventsCount() const
{
  std::stringstream msg;
  msg << "Dropped events count can only be retrieved from a valid, ";
  msg << "complete Event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

void 
detail::ChainData::do_setDroppedEventsCount(unsigned int count)
{
  std::stringstream msg;
  msg << "Dropped events count can only be added to a valid, ";
  msg << "complete Event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32_t detail::ChainData::do_runNumber() const
{
  std::stringstream msg;
  msg << "A run number is only available from a valid, ";
  msg << "complete EVENT or ERROR_EVENT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32_t detail::ChainData::do_lumiSection() const
{
  std::stringstream msg;
  msg << "A luminosity section is only available from a valid, ";
  msg << "complete EVENT or ERROR_EVENT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32_t detail::ChainData::do_eventNumber() const
{
  std::stringstream msg;
  msg << "An event number is only available from a valid, ";
  msg << "complete EVENT or ERROR_EVENT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

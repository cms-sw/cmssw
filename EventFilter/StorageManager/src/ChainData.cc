// $Id: $

#include "toolbox/mem/Reference.h"

#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/StreamID.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Exception.h"

#include "EventFilter/StorageManager/src/ChainData.h"

using namespace stor;

// A ChainData object may or may not contain a Reference.
detail::ChainData::ChainData(toolbox::mem::Reference* pRef) :
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
detail::ChainData::~ChainData()
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

bool detail::ChainData::empty() const
{
  return !_ref;
}

bool detail::ChainData::complete() const
{
  return _complete;
}

bool detail::ChainData::faulty() const
{
  return (_faultyBits != 0);
}

unsigned int detail::ChainData::faultyBits() const
{
  return _faultyBits;
}

bool detail::ChainData::parsable() const
{
  return (_ref) && ((_faultyBits & INVALID_INITIAL_REFERENCE) == 0) &&
    ((_faultyBits & CORRUPT_INITIAL_HEADER) == 0);
}

// this method currently does NOT support operation on empty chains
// (both the existing chain and the new part must be non-empty!)
void detail::ChainData::addToChain(ChainData const& newpart)
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

void detail::ChainData::markComplete()
{
  _complete = true;
}

void detail::ChainData::markFaulty()
{
  _faultyBits |= EXTERNALLY_REQUESTED;
}

void detail::ChainData::markCorrupt()
{
  _faultyBits |= CORRUPT_INITIAL_HEADER;
}

unsigned long* detail::ChainData::getBufferData() const
{
  return _ref 
    ?  static_cast<unsigned long*>(_ref->getDataLocation()) 
    : 0UL;
}

void detail::ChainData::swap(ChainData& other)
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

unsigned long detail::ChainData::totalDataSize() const
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

unsigned long detail::ChainData::dataSize(int fragmentIndex) const
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

unsigned char* detail::ChainData::dataLocation(int fragmentIndex) const
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

unsigned int detail::ChainData::getFragmentID(int fragmentIndex) const
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

unsigned int detail::ChainData::
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

std::string detail::ChainData::hltClassName() const
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

uint32 detail::ChainData::outputModuleId() const
{
  return do_outputModuleId();
}


std::string detail::ChainData::outputModuleLabel() const
{
  return do_outputModuleLabel();
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

uint32 detail::ChainData::hltTriggerCount() const
{
  return do_hltTriggerCount();
}

void
detail::ChainData::hltTriggerBits(std::vector<unsigned char>& bitList) const
{
  do_hltTriggerBits(bitList);
}

void detail::ChainData::assertRunNumber(uint32 runNumber)
{
  do_assertRunNumber(runNumber);
}

uint32 detail::ChainData::runNumber() const
{
  return do_runNumber();
}

uint32 detail::ChainData::lumiSection() const
{
  return do_lumiSection();
}

uint32 detail::ChainData::eventNumber() const
{
  return do_eventNumber();
}

void detail::ChainData::tagForStream(StreamID streamId)
{
  _streamTags.push_back(streamId);
}

void detail::ChainData::tagForEventConsumer(QueueID queueId)
{
  _eventConsumerTags.push_back(queueId);
}

void detail::ChainData::tagForDQMEventConsumer(QueueID queueId)
{
  _dqmEventConsumerTags.push_back(queueId);
}

std::vector<StreamID> const& detail::ChainData::getStreamTags() const
{
  return _streamTags;
}

std::vector<QueueID> const& detail::ChainData::getEventConsumerTags() const
{
  return _eventConsumerTags;
}

std::vector<QueueID> const& detail::ChainData::getDQMEventConsumerTags() const
{
  return _dqmEventConsumerTags;
}

bool
detail::ChainData::validateDataLocation(toolbox::mem::Reference* ref,
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

bool
detail::ChainData::validateMessageSize(toolbox::mem::Reference* ref,
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

bool
detail::ChainData::validateFragmentIndexAndCount(toolbox::mem::Reference* ref,
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

bool
detail::ChainData::validateExpectedFragmentCount(toolbox::mem::Reference* ref,
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

bool
detail::ChainData::validateMessageCode(toolbox::mem::Reference* ref,
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

unsigned long detail::ChainData::do_headerSize() const
{
  return 0;
}

unsigned char* detail::ChainData::do_headerLocation() const
{
  return 0;
}

unsigned char*
detail::ChainData::do_fragmentLocation(unsigned char* dataLoc) const
{
  return dataLoc;
}

uint32 detail::ChainData::do_outputModuleId() const
{
  std::stringstream msg;
  msg << "An output module ID is only available from a valid, ";
  msg << "complete INIT or Event message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

std::string detail::ChainData::do_outputModuleLabel() const
{
  std::stringstream msg;
  msg << "An output module label is only available from a valid, ";
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

uint32 detail::ChainData::do_hltTriggerCount() const
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

void detail::ChainData::do_assertRunNumber(uint32 runNumber)
{}

uint32 detail::ChainData::do_runNumber() const
{
  std::stringstream msg;
  msg << "A run number is only available from a valid, ";
  msg << "complete EVENT or ERROR_EVENT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32 detail::ChainData::do_lumiSection() const
{
  std::stringstream msg;
  msg << "A luminosity section is only available from a valid, ";
  msg << "complete EVENT or ERROR_EVENT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

uint32 detail::ChainData::do_eventNumber() const
{
  std::stringstream msg;
  msg << "An event number is only available from a valid, ";
  msg << "complete EVENT or ERROR_EVENT message.";
  XCEPT_RAISE(stor::exception::WrongI2OMessageType, msg.str());
}

/**
 * The DQMEventMsgView class is used to view the DQM data messages that
 * are exchanged between the filter units and the storage manager.
 *
 * 09-Feb-2007 - Initial Implementation
 */

#include "IOPool/Streamer/interface/DQMEventMessage.h"
#include "FWCore/Utilities/interface/Exception.h"

#define MAX_STRING_SIZE 10000

/**
 * Constructor.
 */
DQMEventMsgView::DQMEventMsgView(void* buf):
  buf_((uint8*)buf),head_(buf)
{
  uint8* bufPtr;
  uint32 len;

  // verify that the buffer actually contains a DQM Event message
  if (this->code() != Header::DQM_EVENT)
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Invalid DQM Event message code (" << this->code()
        << "). Should be " << Header::DQM_EVENT << "\n";
    }

  // verify that the message has a protocol that we support
  if (this->protocolVersion() != 2)
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Unsupport protocol version (" << this->protocolVersion() << ").\n";
    }

  // set our buffer pointer to just beyond the fixed header data
  bufPtr = buf_ + sizeof(DQMEventHeader);

  // determine the release tag
  len = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  if (len >= 0)
    {
      if (len <= MAX_STRING_SIZE) // prevent something totally crazy
        {
          releaseTag_.append((char *) bufPtr, len);
        }
      bufPtr += len;
    }

  // determine the top-level folder name
  len = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  if (len >= 0)
    {
      if (len <= MAX_STRING_SIZE) // prevent something totally crazy
        {
          folderName_.append((char *) bufPtr, len);
        }
      bufPtr += len;
    }

  // determine the number of subfolders
  subFolderCount_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);

  // loop over the subfolders to extract relevant quantities
  nameListPtr_.reset(new std::vector<std::string>());
  for (int idx = 0; idx < (int) subFolderCount_; idx++)
    {
      // number of MEs in subfolder
      uint32 meCount = convert32(bufPtr);
      bufPtr += sizeof(uint32);
      meCountList_.push_back(meCount);

      // subfolder name
      std::string subFolderName = "Subfolder " + idx;
      uint32 nameLen = convert32(bufPtr);
      bufPtr += sizeof(uint32);
      if (nameLen >= 0)
        {
          if (nameLen <= MAX_STRING_SIZE) // prevent something totally crazy
            {
              subFolderName.clear();
              subFolderName.append((char *) bufPtr, nameLen);
            }
          bufPtr += nameLen;
        }
      nameListPtr_->push_back(subFolderName);
      subFolderIndexTable_[subFolderName] = idx;
    }

  // determine the event length and address
  eventLen_ = convert32(bufPtr);
  bufPtr += sizeof(uint32);
  eventAddr_ = bufPtr;

  // check that the event data doesn't extend beyond the reported
  // size of the message
  if ((this->headerSize() + this->eventLength()) > this->size())
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Inconsistent data sizes. The size of the header ("
        << this->headerSize() << ") and the data (" << this->eventLength()
        << ") exceed the size of the message (" << this->size() << ").\n";
    }
}

/**
 * Returns a shared pointer to the list of subfolder names.
 */
boost::shared_ptr< std::vector<std::string> >
    DQMEventMsgView::subFolderNames() const
{
  return nameListPtr_;
}

/**
 * Returns the name of the subfolder at the specified index.
 */
std::string DQMEventMsgView::subFolderName(uint32 const subFolderIndex) const
{
  // catch attempts to access an invalid entry
  if (subFolderIndex >= subFolderCount_)
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Invalid subfolder index (" << subFolderIndex << ") - "
        << "the number of subfolders is " << subFolderCount_ << ".\n";
    }

  return (*nameListPtr_)[subFolderIndex];
}

/**
 * Returns the number of monitor elements in the specified subfolder.
 */
uint32 DQMEventMsgView::meCount(std::string const& subFolderName) const
{
  // lookup the index of the specified subfolder
  std::map<std::string, uint32>::const_iterator subFolderIter;
  subFolderIter = subFolderIndexTable_.find(subFolderName);

  // throw an exception if the name was not found
  if (subFolderIter == subFolderIndexTable_.end())
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Unable to find the subfolder index for \""
        << subFolderName << "\".\n";
    }

  // fetch the count by index
  return this->meCount(subFolderIter->second);
}

/**
 * Returns the number of monitor elements in the subfolder at the
 * specified index.
 */
uint32 DQMEventMsgView::meCount(uint32 const subFolderIndex) const
{
  // catch attempts to access an invalid entry
  if (subFolderIndex >= subFolderCount_)
    {
      throw cms::Exception("MessageDecoding", "DQMEventMsgView")
        << "Invalid subfolder index (" << subFolderIndex << ") - "
        << "the number of subfolders is " << subFolderCount_ << ".\n";
    }

  return meCountList_[subFolderIndex];
}

/**
 * Returns the protocol version of the DQM Event.
 */
uint32 DQMEventMsgView::protocolVersion() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->protocolVersion_);
}

/**
 * Returns the size of the DQM Event header.
 */
uint32 DQMEventMsgView::headerSize() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->headerSize_);
}

/**
 * Returns the run number associated with the DQM Event.
 */
uint32 DQMEventMsgView::runNumber() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->runNumber_);
}

/**
 * Returns the event number associated with the DQM Event.
 */
uint32 DQMEventMsgView::eventNumberAtUpdate() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->eventNumber_);
}

/**
 * Returns the lumi section associated with the DQM Event.
 */
uint32 DQMEventMsgView::lumiSection() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->lumiSection_);
}

/**
 * Returns the update number of the DQM Event.
 */
uint32 DQMEventMsgView::updateNumber() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->updateNumber_);
}

/**
 * Returns the compression flag (uncompressed data size or zero if uncompressed).
 */
uint32 DQMEventMsgView::compressionFlag() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->compressionFlag_);
}

/**
 * Returns the process ID of the filter unit that created this update.
 */
uint32 DQMEventMsgView::fuProcessId() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->fuProcessId_);
}

/**
 * Returns the GUID of the filter unit that created this update.
 */
uint32 DQMEventMsgView::fuGuid() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->fuGuid_);
}

/**
 * Returns the reserved word associated with the DQM Event.
 */
uint32 DQMEventMsgView::reserved() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return convert32(h->reserved_);
}

//uint64 DQMEventMsgView::timeStamp() const
edm::Timestamp DQMEventMsgView::timeStamp() const
{
  DQMEventHeader* h = (DQMEventHeader*)buf_;
  return edm::Timestamp(convert64(h->timeStamp_));
}


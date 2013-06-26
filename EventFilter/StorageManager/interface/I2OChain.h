// $Id: I2OChain.h,v 1.16 2012/04/20 10:48:18 mommsen Exp $
/// @file: I2OChain.h 

#ifndef EventFilter_StorageManager_I2OChain_h
#define EventFilter_StorageManager_I2OChain_h

#include <vector>

#include "boost/shared_ptr.hpp"
#include "toolbox/mem/Reference.h"

#include "EventFilter/StorageManager/interface/FragKey.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/StreamID.h"
#include "EventFilter/StorageManager/interface/Utils.h"

#include "IOPool/Streamer/interface/MsgTools.h"

namespace stor {

  class DQMKey;


  /**
   * List of one or multiple I2O messages representing event fragments. 
   *
   * It wraps several toolbox::mem::Reference chained together and 
   * assures that the corresponding release methods are called when 
   * the last instance of I2OChain goes out of scope.
   *
   * $Author: mommsen $
   * $Revision: 1.16 $
   * $Date: 2012/04/20 10:48:18 $
   */


  // We need only declare ChainData here; it is defined in I2OChain.cc.
  namespace detail
  {
    class ChainData;
  }

  class I2OChain
  {
  public:


    /**
       A default-constructed I2OChain manages no Reference.
    */
    I2OChain();

    /**
       Create an I2OChain that will manage the Reference at address
       pRef, and assure that release is called on it only once,
       regardless of how many copies of the I2OChain object have been
       made.
       NOTE that if the message fragment contained in the XDAQ
       buffer is corrupted in some way, the chain will be marked "faulty"
       and should not be used in normal fragment processing.
    */
    explicit I2OChain(toolbox::mem::Reference* pRef);

    /**
       A copy of an I2OChain shares management of the underlying
       Reference. We avoid calling Reference::duplicate.
     */
    I2OChain(I2OChain const& other);

    /**
       Destroying an I2OChain does not release the managed Reference,
       unless that I2OChain is the last one managing that Reference.
    */
    ~I2OChain();

    /**
       Assigning to an I2OChain causes the left-hand side of the
       assignment to relinquish management of any Reference it might
       have had. If the left-hand side was the only chain managing
       that Reference, it will be released. After the assignment, the
       left-hand side shares management of the underlying Reference of
       the right-hand side.
     */
    I2OChain& operator=(I2OChain const& rhs);

    /**
       Standard swap.
     */
    void swap(I2OChain& other);

    /**
     * Returns true if there is no Reference managed by *this.
     */
    bool empty() const;

    /**
     * Returns true if all fragments of an event are available
     */
    bool complete() const;

    /**
     * Returns true if the chain has been marked faulty (the internal
     * data does not represent a valid message fragment or the fragments
     * in the chain do not represent a complete, valid message).
     */
    bool faulty() const;

    /**
     * Returns a bitmask containing the faulty bits.
     */
    unsigned int faultyBits() const;

    /**
       Adds fragments from another chain to the current chain taking
       care that all fragments are chained in the right order. This
       destructively modifies newpart so that it no longer is part of
       managing any Reference: newpart is made empty.

       If newpart contains chain elements that do not have the same
       fragment key (FragKey) as the current chain, an exception is thrown.

       If newpart contains chain elements that already appear to exist
       in the current chain (e.g. fragment number 3 is added a second time),
       no exceptions are thrown and newpart is made empty, but the current
       chain is marked as faulty.
     */
    void addToChain(I2OChain& newpart);

    /**
       Mark this chain as known to be complete.
     */
    //void markComplete();

    /**
       Mark this chain as known to be faulty.  The failure modes that
       result in a chain being marked faulty include chains that have
       duplicate fragments and chains that never become complete after
       a timeout interval.
     */
    void markFaulty();

    /**
       Return the address at which the data in buffer managed by the
       Reference begins. If the chain is empty, a null pointer is
       returned.
    */
    unsigned long* getBufferData() const;


    /**
       Abandon management of the managed Reference, if there is
       one. After this call, *this will be in the same state as if it
       had been default-constructed.
     */
    void release();

    /**
       Returns the time when the I2OChain was created. This time corresponds
       to the time when the first fragment of the I2OChain was added.

       The value corresponds to the number of seconds since the epoch
       (including a fractional part good to the microsecond level).
       A negative value indicates that an error occurred when fetching 
       the time from the operating system.
     */
    utils::TimePoint_t creationTime() const;

    /**
       Returns the time when the last fragment was added to the I2OChain.

       The value corresponds to the number of seconds since the epoch
       (including a fractional part good to the microsecond level).
       A negative value indicates that an error occurred when fetching 
       the time from the operating system.
     */
    utils::TimePoint_t lastFragmentTime() const;

    /**
       Returns the stale window start time.  This is the time that
       should be used when checking if a chain has become stale.

       The value corresponds to the number of seconds since the epoch
       (including a fractional part good to the microsecond level).
       A negative value indicates that an error occurred when fetching 
       the time from the operating system.
     */
    utils::TimePoint_t staleWindowStartTime() const;

    /**
       Add the Duration_t in seconds to the stale window start time 
     */
    void addToStaleWindowStartTime(const utils::Duration_t);

    /**
       Sets the stale window start time to "now".
     */
    void resetStaleWindowStartTime();

    /**
       Tags the chain with the specified disk writing stream ID.  This
       means that the data in the chain should be sent to the specified
       disk stream.
     */
    void tagForStream(StreamID);

    /**
       Tags the chain with the specified event consumer queue ID.  This
       means that the data in the chain should be sent to the specified
       event consumer queue.
     */
    void tagForEventConsumer(QueueID);

    /**
       Tags the chain with the specified DQM event consumer queue ID.  This
       means that the data in the chain should be sent to the specified
       DQM event consumer queue.
     */
    void tagForDQMEventConsumer(QueueID);

    /**
       Returns true if the chain has been tagged for any disk stream
       and false otherwise.
    */
    bool isTaggedForAnyStream() const;

    /**
       Returns true if the chain has been tagged for any event consumer
       and false otherwise.
    */
    bool isTaggedForAnyEventConsumer() const;

    /**
       Returns true if the chain has been tagged for any DQM event consumer
       and false otherwise.
    */
    bool isTaggedForAnyDQMEventConsumer() const;

    /**
       Returns the list of disk streams (stream IDs) that
       this chain has been tagged for.
       An empty list is returned if the chain is empty.
       NOTE that this method returns a copy of the list, so it
       should only be used for testing which streams have been tagged,
       *not* for for modifying the list of tags.
    */
    std::vector<StreamID> getStreamTags() const;

    /**
       Returns the list of event consumers (queue IDs) that
       this chain has been tagged for.
       An empty list is returned if the chain is empty.
       NOTE that this method returns a copy of the list, so it
       should only be used for testing which consumers have been tagged,
       *not* for for modifying the list of tags.
    */
    QueueIDs getEventConsumerTags() const;

    /**
       Returns the list of DQM event consumers (queue IDs) that
       this chain has been tagged for.
       An empty list is returned if the chain is empty.
       NOTE that this method returns a copy of the list, so it
       should only be used for testing which consumers have been tagged,
       *not* for for modifying the list of tags.
    */
    QueueIDs getDQMEventConsumerTags() const;

    /**
      Return the number of dropped events found in the EventHeader
     */
    unsigned int droppedEventsCount() const;

    /**
        Add the number of dropped (skipped) events to the EVENT message
        header.
    */
    void setDroppedEventsCount(unsigned int);

    /**
       Returns the message code for the chain. Valid values
       are Header::INVALID, Header::INIT, Header::EVENT, Header::DQM_EVENT,
       and Header::ERROR_EVENT from IOPool/Streamer/interface/MsgHeader.h.
     */
    unsigned int messageCode() const;

    /**
       Returns the I2O function code for the chain. Valid values
       are defined in interface/shared/i2oXFunctionCodes.h.
       If now chain is found, 0xffff is returned.
     */
    unsigned short i2oMessageCode() const;

    /**
       Returns the resource broker buffer ID from the contained message.
       If no valid buffer ID can be determined, zero is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    unsigned int rbBufferId() const;

    /**
       Returns the HLT local ID from the contained message.
       If no valid local ID can be determined, zero is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    unsigned int hltLocalId() const;

    /**
       Returns the HLT instance number from the contained message.
       If no valid instance number can be determined, zero is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    unsigned int hltInstance() const;

    /**
       Returns the HLT TID from the contained message.
       If no valid TID can be determined, zero is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    unsigned int hltTid() const;

    /**
       Returns the HLT URL from the contained message.  If no
       valid URL can be determined, an empty string is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    std::string hltURL() const;

    /**
       Returns the HLT class name from the contained message.  If no
       valid class name can be determined, an empty string is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    std::string hltClassName() const;

    /**
       Returns the filter unit process ID from the contained message.
       If no valid process ID can be determined, zero is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    unsigned int fuProcessId() const;

    /**
       Returns the filter unit GUID from the contained message.
       If no valid GUID can be determined, zero is returned.
       NOTE that you must test if messageCode() != Header::INVALID to
       determine that the returned value is valid.
     */
    unsigned int fuGuid() const;

    /**
       Returns the fragment key for the chain.  The fragment key
       is the entity that uniquely identifies all of the fragments
       from a particular event.
     */
    FragKey fragmentKey() const;


    /**
       Returns the total memory occupied by all message fragments in
       the chain. This includes all I2O headers and
     */
    size_t memoryUsed() const;

    /**
       Returns the number of frames currently contained in the chain.
       NOTE that this will be less than the total number of expected frames
       before the chain is complete and equal to the total number of expected
       frames after the chain is complete.  And it can be zero if the
       chain is empty.  If the chain is faulty, this method still returns
       the number of fragments contained in the chain, but those fragments
       may not correspond to a valid I2O message.
     */
    unsigned int fragmentCount() const;

    /**
       Returns the total size of all of the contained message fragments.
       For chains that are marked complete, this is the size of the actual
       INIT, EVENT, DQM_EVENT, or ERROR_EVENT message contained in the chain.
       For incomplete chains, this method will return an invalid value.
       If the chain has been marked as "faulty", this method will
       return the total size of the data that will be returned from
       the dataSize() method on each of the fragments, even though
       those fragments may not correspond to actual storage manager messages.
     */
    unsigned long totalDataSize() const;

    /**
       Returns the size of the specified message fragment
       (indexed from 0 to N-1, where N is the total number of fragments).
       For complete chains, this method returns the sizes of the actual
       INIT, EVENT, ERROR_EVENT, or DQM_EVENT message fragments.
       If the chain has been marked as "faulty", this method will still
       return a valid data size for all fragment indices.  However,
       in that case, the sizes may not correspond to the underlying
       INIT, EVENT, etc. message.
     */
    unsigned long dataSize(int fragmentIndex) const;

    /**
       Returns the start address of the specified message fragment
       (indexed from 0 to N-1, where N is the total number of fragments).
       For complete chains, this method returns pointers to the actual
       INIT, EVENT, DQM_EVENT, or ERROR_EVENT message fragments.
       If the chain has been marked as "faulty", this method will still
       return a valid data location for all fragment indices.  However,
       in that case, the data may not correspond to an underlying
       INIT, EVENT, etc. message.
     */
    unsigned char* dataLocation(int fragmentIndex) const;

    /**
       Returns the fragmentID of the specified message fragment
       (indexed from 0 to N-1, where N is the total number of fragments).
       This value varies from 0 to N-1 and should match the fragment index,
       so this method is probably only useful for testing.
     */
    unsigned int getFragmentID(int fragmentIndex) const;

    /**
       Returns the size of the header part of the message that is contained
       in the chain.  For complete chains of type INIT, EVENT, and
       DQM_EVENT, this method returns the true size of the message header.
       For other types of chains, and for chains that have been marked
       "faulty", this method will return a size of zero.
     */
    unsigned long headerSize() const;

    /**
       Returns the start address of the header part of the message that is
       contained in the chain.  For complete chains of type INIT, EVENT, and
       DQM_EVENT, this method returns the true pointer to the message header.
       For other types of chains, and for chains that have been marked
       "faulty", this method will return a NULL pointer.
     */
    unsigned char* headerLocation() const;

    /**
       Copies the internally managed fragments to the specified
       vector in one contiguous set.  Note that *NO* tests are done by
       this method - it can be run on empty, incomplete, faulty, and
       complete chains - and the result could be an incomplete or faulty
       storage manager message or worse.  If the I2O fragments in the chain
       are corrupt, the data copied into the buffer could be the raw
       I2O messages, including headers.
       Returns the number of bytes copied.
     */
    unsigned int copyFragmentsIntoBuffer(std::vector<unsigned char>& buff) const;

    /**
       Returns the output module label contained in the message, if and
       only if, the message is an INIT message.  Otherwise,
       an exception is thrown.
     */
    std::string outputModuleLabel() const;

    /**
       Returns the output module ID contained in the message, if and
       only if, the message is an INIT or an Event message.  Otherwise,
       an exception is thrown.
     */
    uint32_t outputModuleId() const;

    /**
       Returns the number of slave EPs reported in the message, if and
       only if, the message is an INIT message.  Otherwise,
       an exception is thrown.
     */
    uint32_t nExpectedEPs() const;

    /**
       Returns the top folder contained in the message, if and
       only if, the message is a DQM event message.  Otherwise,
       an exception is thrown.
     */
    std::string topFolderName() const;

    /**
       Returns the DQM key constructed from the message, if and
       only if, the message is a DQM event message.  Otherwise,
       an exception is thrown. The DQM key uniquely identifies
       DQM events to be collated.
     */
    DQMKey dqmKey() const;

    /**
       Copies the HLT trigger names into the specified vector, if and
       only if, the message is an INIT message.  Otherwise,
       an exception is thrown.
     */
    void hltTriggerNames(Strings& nameList) const;

    /**
       Copies the HLT trigger names into the specified vector, if and
       only if, the message is an INIT message.  Otherwise,
       an exception is thrown.
     */
    void hltTriggerSelections(Strings& nameList) const;

    /**
       Copies the L1 trigger names into the specified vector, if and
       only if, the message is an INIT message.  Otherwise,
       an exception is thrown.
     */
    void l1TriggerNames(Strings& nameList) const;

    /**
       Returns the number HLT trigger bits contained in the message, if
       and only if, the message is an Event message.  Otherwise,
       an exception is thrown.
     */
    uint32_t hltTriggerCount() const;

    /**
       Copies the HLT trigger bits into the specified vector, if and
       only if, the message is an Event message.  Otherwise,
       an exception is thrown.  The vector will be resized so that
       it contains the full number of HLT bits (given by the
       hltCount() method) with two bits per HLT trigger.
     */
    void hltTriggerBits(std::vector<unsigned char>& bitList) const;

    /**
       Returns the run number of the message, if and only if, the 
       message is an Event or ErrorEvent message. 
       Otherwise an exception is thrown.
     */
    uint32_t runNumber() const;

    /**
       Returns the luminosity section of the message, if and only if,
       the message is an Event or ErrorEvent message. 
       Otherwise an exception is thrown.
     */
    uint32_t lumiSection() const;

    /**
       Returns the event number of the message, if and only if, the 
       message is an Event or ErrorEvent message. 
       Otherwise an exception is thrown.
     */
    uint32_t eventNumber() const;

    /**
       Returns the adler32 checksum as found in the message if available.
       Otherwise 0 is returned.
     */
    uint32_t adler32Checksum() const;

    /**
       Checks that the run number found in the I2OChain header
       corresponds to the run number given as argument.
       It throws stor::exception::RunNumberMismatch if it is 
       not the case.
       For error events, the given run number will be used by
       the StorageManager, but it will *not* be changed in the I2O header.
     */
    void assertRunNumber(uint32_t runNumber);

    /**
       Returns true if the I2O function code indicates that the message
       represents an end-of-lumi-section signal.
     */
    bool isEndOfLumiSectionMessage() const;

  private:

    boost::shared_ptr<detail::ChainData> data_;
  };
  
} // namespace stor

#endif // EventFilter_StorageManager_I2OChain_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

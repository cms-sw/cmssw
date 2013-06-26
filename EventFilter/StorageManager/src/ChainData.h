// $Id: ChainData.h,v 1.20 2012/04/20 10:48:01 mommsen Exp $
/// @file: ChainData.h

#ifndef CHAINDATA_H
#define CHAINDATA_H

#include "toolbox/mem/Reference.h"

#include "IOPool/Streamer/interface/MsgHeader.h"
#include "IOPool/Streamer/interface/MsgTools.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/FragKey.h"
#include "EventFilter/StorageManager/interface/StreamID.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "EventFilter/StorageManager/interface/Exception.h"


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
    protected:
      enum BitMasksForFaulty { INVALID_INITIAL_REFERENCE = 0x1,
                               CORRUPT_INITIAL_HEADER = 0x2,
                               INVALID_SECONDARY_REFERENCE = 0x4,
                               CORRUPT_SECONDARY_HEADER = 0x8,
                               TOTAL_COUNT_MISMATCH = 0x10,
                               FRAGMENTS_OUT_OF_ORDER = 0x20,
                               DUPLICATE_FRAGMENT = 0x40,
                               INCOMPLETE_MESSAGE = 0x80,
                               WRONG_CHECKSUM = 0x100,
                               EXTERNALLY_REQUESTED = 0x10000 };


    public:
      explicit ChainData(unsigned short i2oMessageCode = 0x9999,
                         unsigned int messageCode = Header::INVALID);
      virtual ~ChainData();
      bool empty() const;
      bool complete() const;
      bool faulty() const;
      unsigned int faultyBits() const;
      bool parsable() const;
      bool headerOkay() const;
      void addFirstFragment(toolbox::mem::Reference*);
      void addToChain(ChainData const&);
      void markComplete();
      void markFaulty();
      void markCorrupt();
      unsigned long* getBufferData() const;
      void swap(ChainData& other);
      unsigned int messageCode() const {return messageCode_;}
      unsigned short i2oMessageCode() const {return i2oMessageCode_;}
      FragKey const& fragmentKey() const {return fragKey_;}
      unsigned int fragmentCount() const {return fragmentCount_;}
      unsigned int rbBufferId() const {return rbBufferId_;}
      unsigned int hltLocalId() const {return hltLocalId_;}
      unsigned int hltInstance() const {return hltInstance_;}
      unsigned int hltTid() const {return hltTid_;}
      unsigned int fuProcessId() const {return fuProcessId_;}
      unsigned int fuGuid() const {return fuGuid_;}
      utils::TimePoint_t creationTime() const {return creationTime_;}
      utils::TimePoint_t lastFragmentTime() const {return lastFragmentTime_;}
      utils::TimePoint_t staleWindowStartTime() const {return staleWindowStartTime_;}
      void addToStaleWindowStartTime(const utils::Duration_t duration) {
        staleWindowStartTime_ += duration;
      }
      void resetStaleWindowStartTime() {
        staleWindowStartTime_ = utils::getCurrentTime();
      }
      unsigned int droppedEventsCount() const;
      void setDroppedEventsCount(unsigned int);
      size_t memoryUsed() const;
      unsigned long totalDataSize() const;
      unsigned long dataSize(int fragmentIndex) const;
      unsigned char* dataLocation(int fragmentIndex) const;
      unsigned int getFragmentID(int fragmentIndex) const;
      unsigned int copyFragmentsIntoBuffer(std::vector<unsigned char>& buff) const;

      unsigned long headerSize() const;
      unsigned char* headerLocation() const;

      std::string hltURL() const;
      std::string hltClassName() const;
      uint32_t outputModuleId() const;
      uint32_t nExpectedEPs() const;

      std::string outputModuleLabel() const;
      void hltTriggerNames(Strings& nameList) const;
      void hltTriggerSelections(Strings& nameList) const;
      void l1TriggerNames(Strings& nameList) const;

      void assertRunNumber(uint32_t runNumber);

      uint32_t runNumber() const;
      uint32_t lumiSection() const;
      uint32_t eventNumber() const;
      uint32_t adler32Checksum() const;

      std::string topFolderName() const;
      DQMKey dqmKey() const;

      uint32_t hltTriggerCount() const;
      void hltTriggerBits(std::vector<unsigned char>& bitList) const;

      void tagForStream(StreamID);
      void tagForEventConsumer(QueueID);
      void tagForDQMEventConsumer(QueueID);
      bool isTaggedForAnyStream() const {return !streamTags_.empty();}
      bool isTaggedForAnyEventConsumer() const {return !eventConsumerTags_.empty();}
      bool isTaggedForAnyDQMEventConsumer() const {return !dqmEventConsumerTags_.empty();}
      std::vector<StreamID> const& getStreamTags() const;
      QueueIDs const& getEventConsumerTags() const;
      QueueIDs const& getDQMEventConsumerTags() const;

      bool isEndOfLumiSectionMessage() const;

    private:
      std::vector<StreamID> streamTags_;
      QueueIDs eventConsumerTags_;
      QueueIDs dqmEventConsumerTags_;

      utils::TimePoint_t creationTime_;
      utils::TimePoint_t lastFragmentTime_;
      utils::TimePoint_t staleWindowStartTime_;

      void checkForCompleteness();
      bool validateAdler32Checksum();
      uint32_t calculateAdler32() const;

    protected:
      toolbox::mem::Reference* ref_;

      bool complete_;
      unsigned int faultyBits_;

      unsigned int messageCode_;
      unsigned short i2oMessageCode_;
      FragKey fragKey_;
      unsigned int fragmentCount_;
      unsigned int expectedNumberOfFragments_;
      unsigned int rbBufferId_;
      unsigned int hltLocalId_;
      unsigned int hltInstance_;
      unsigned int hltTid_;
      unsigned int fuProcessId_;
      unsigned int fuGuid_;

      inline bool validateDataLocation(
        toolbox::mem::Reference* ref,
        BitMasksForFaulty maskToUse
      );
      inline bool validateMessageSize(
        toolbox::mem::Reference* ref,
        BitMasksForFaulty maskToUse
      );
      inline bool validateFragmentIndexAndCount(
        toolbox::mem::Reference* ref,
        BitMasksForFaulty maskToUse
      );
      inline bool validateExpectedFragmentCount(
        toolbox::mem::Reference* ref,
        BitMasksForFaulty maskToUse
      );
      inline bool validateFragmentOrder(
        toolbox::mem::Reference* ref,
        int& indexValue
      );
      inline bool validateMessageCode(
        toolbox::mem::Reference* ref,
        unsigned short expectedI2OMessageCode
      );

      virtual inline size_t do_i2oFrameSize() const { return 0; }
      virtual inline unsigned long do_headerSize() const { return 0; }
      virtual inline unsigned char* do_headerLocation() const { return 0; }
      virtual inline unsigned char* do_fragmentLocation(unsigned char* dataLoc) const { return dataLoc; }
      virtual inline uint32_t do_adler32Checksum() const { return 0; }

      virtual std::string do_outputModuleLabel() const;
      virtual uint32_t do_outputModuleId() const;
      virtual uint32_t do_nExpectedEPs() const;
      virtual void do_hltTriggerNames(Strings& nameList) const;
      virtual void do_hltTriggerSelections(Strings& nameList) const;
      virtual void do_l1TriggerNames(Strings& nameList) const;
      virtual unsigned int do_droppedEventsCount() const;
      virtual void do_setDroppedEventsCount(unsigned int);

      virtual std::string do_topFolderName() const;
      virtual DQMKey do_dqmKey() const;

      virtual uint32_t do_hltTriggerCount() const;
      virtual void do_hltTriggerBits(std::vector<unsigned char>& bitList) const;

      virtual inline void do_assertRunNumber(uint32_t runNumber) {};

      virtual uint32_t do_runNumber() const;
      virtual uint32_t do_lumiSection() const;
      virtual uint32_t do_eventNumber() const;

    }; // class ChainData


    //////////////////////
    //// InitMsgData: ////
    //////////////////////

    class InitMsgData : public ChainData
    {

    public:

      explicit InitMsgData(toolbox::mem::Reference* pRef);
      ~InitMsgData() {}

    protected:

      inline size_t do_i2oFrameSize() const;
      inline unsigned long do_headerSize() const;
      inline unsigned char* do_headerLocation() const;
      inline unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      inline uint32_t do_adler32Checksum() const;

      std::string do_outputModuleLabel() const;
      uint32_t do_outputModuleId() const;
      uint32_t do_nExpectedEPs() const { return nExpectedEPs_; };
      void do_hltTriggerNames(Strings& nameList) const;
      void do_hltTriggerSelections(Strings& nameList) const;
      void do_l1TriggerNames(Strings& nameList) const;

    private:

      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool headerFieldsCached_;
      mutable std::vector<unsigned char> headerCopy_;
      mutable unsigned long headerSize_;
      mutable unsigned char* headerLocation_;
      mutable uint32_t adler32_;
      mutable uint32_t outputModuleId_;
      mutable uint32_t nExpectedEPs_;
      mutable std::string outputModuleLabel_;
      mutable Strings hltTriggerNames_;
      mutable Strings hltTriggerSelections_;
      mutable Strings l1TriggerNames_;

    }; // class InitMsgData


    ///////////////////////
    //// EventMsgData: ////
    ///////////////////////

    class EventMsgData : public ChainData
    {

    public:

      explicit EventMsgData(toolbox::mem::Reference* pRef);
      ~EventMsgData() {}

    protected:

      inline size_t do_i2oFrameSize() const;
      inline unsigned long do_headerSize() const;
      inline unsigned char* do_headerLocation() const;
      inline unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      inline uint32_t do_adler32Checksum() const;

      uint32_t do_outputModuleId() const;
      uint32_t do_hltTriggerCount() const;
      void do_hltTriggerBits(std::vector<unsigned char>& bitList) const;
      void do_assertRunNumber(uint32_t runNumber);
      uint32_t do_runNumber() const;
      uint32_t do_lumiSection() const;
      uint32_t do_eventNumber() const;

      unsigned int do_droppedEventsCount() const;
      void do_setDroppedEventsCount(unsigned int);

    private:

      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool headerFieldsCached_;
      mutable std::vector<unsigned char> headerCopy_;
      mutable unsigned long headerSize_;
      mutable unsigned char* headerLocation_;
      mutable uint32_t outputModuleId_;
      mutable uint32_t hltTriggerCount_;
      mutable std::vector<unsigned char> hltTriggerBits_;
      mutable uint32_t runNumber_;
      mutable uint32_t lumiSection_;
      mutable uint32_t eventNumber_;
      mutable uint32_t adler32_;
      mutable unsigned int droppedEventsCount_;

    }; // EventMsgData


    /////////////////////////
    //// DQMEventMsgData ////
    /////////////////////////

    class DQMEventMsgData : public ChainData
    {

    public:

      explicit DQMEventMsgData(toolbox::mem::Reference* pRef);
      ~DQMEventMsgData() {}

    protected:

      inline size_t do_i2oFrameSize() const;
      inline unsigned long do_headerSize() const;
      inline unsigned char* do_headerLocation() const;
      inline unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      inline uint32_t do_adler32Checksum() const;

      std::string do_topFolderName() const;
      DQMKey do_dqmKey() const;
      inline void do_assertRunNumber(uint32_t runNumber);
      uint32_t do_runNumber() const;
      uint32_t do_lumiSection() const;

    private:

      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool headerFieldsCached_;
      mutable std::vector<unsigned char> headerCopy_;
      mutable unsigned long headerSize_;
      mutable unsigned char* headerLocation_;
      mutable std::string topFolderName_;
      mutable DQMKey dqmKey_;
      mutable uint32_t adler32_;

    }; // class DQMEventMsgData


    ///////////////////////////
    //// ErrorEventMsgData ////
    ///////////////////////////

    class ErrorEventMsgData : public ChainData
    {

    public:

      explicit ErrorEventMsgData(toolbox::mem::Reference* pRef);
      ~ErrorEventMsgData() {}

    protected:

      inline size_t do_i2oFrameSize() const;
      inline unsigned long do_headerSize() const;
      inline unsigned char* do_headerLocation() const;
      inline unsigned char* do_fragmentLocation(unsigned char* dataLoc) const;
      inline void do_assertRunNumber(uint32_t runNumber);
      uint32_t do_runNumber() const;
      uint32_t do_lumiSection() const;
      uint32_t do_eventNumber() const;

    private:

      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool headerFieldsCached_;
      mutable std::vector<unsigned char> headerCopy_;
      mutable unsigned long headerSize_;
      mutable unsigned char* headerLocation_;
      mutable uint32_t runNumber_;
      mutable uint32_t lumiSection_;
      mutable uint32_t eventNumber_;

    }; // class ErrorEventMsgData


    /////////////////////////////
    //// EndLumiSectMsgData: ////
    /////////////////////////////

    class EndLumiSectMsgData : public ChainData
    {

    public:

      explicit EndLumiSectMsgData( toolbox::mem::Reference* pRef );
      ~EndLumiSectMsgData() {}

    protected:

      inline uint32_t do_runNumber() const { return runNumber_; }
      inline uint32_t do_lumiSection() const { return lumiSection_; }

    private:

      mutable uint32_t runNumber_;
      mutable uint32_t lumiSection_;

    }; // class EndLumiSectMsgData


  } // namespace detail

} // namespace stor

#endif

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -

// $Id: ChainData.h,v 1.13 2010/05/17 15:59:09 mommsen Exp $
/// @file: ChainData.h

#ifndef CHAINDATA_H
#define CHAINDATA_H

#include "toolbox/mem/Reference.h"

#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

#include "EventFilter/Utilities/interface/i2oEvfMsgs.h"

#include "EventFilter/StorageManager/interface/Utils.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
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
      unsigned int messageCode() const {return _messageCode;}
      unsigned short i2oMessageCode() const {return _i2oMessageCode;}
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
      bool isTaggedForAnyStream() const {return !_streamTags.empty();}
      bool isTaggedForAnyEventConsumer() const {return !_eventConsumerTags.empty();}
      bool isTaggedForAnyDQMEventConsumer() const {return !_dqmEventConsumerTags.empty();}
      std::vector<StreamID> const& getStreamTags() const;
      std::vector<QueueID> const& getEventConsumerTags() const;
      std::vector<QueueID> const& getDQMEventConsumerTags() const;

      bool isEndOfLumiSectionMessage() const;

    private:
      std::vector<StreamID> _streamTags;
      std::vector<QueueID> _eventConsumerTags;
      std::vector<QueueID> _dqmEventConsumerTags;

      utils::time_point_t _creationTime;
      utils::time_point_t _lastFragmentTime;
      utils::time_point_t _staleWindowStartTime;

      void checkForCompleteness();
      bool validateAdler32Checksum();
      uint32_t calculateAdler32() const;

    protected:
      toolbox::mem::Reference* _ref;

      bool _complete;
      unsigned int _faultyBits;

      unsigned int _messageCode;
      unsigned short _i2oMessageCode;
      FragKey _fragKey;
      unsigned int _fragmentCount;
      unsigned int _expectedNumberOfFragments;
      unsigned int _rbBufferId;
      unsigned int _hltLocalId;
      unsigned int _hltInstance;
      unsigned int _hltTid;
      unsigned int _fuProcessId;
      unsigned int _fuGuid;

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

      virtual uint32_t do_outputModuleId() const;
      virtual std::string do_outputModuleLabel() const;
      virtual void do_hltTriggerNames(Strings& nameList) const;
      virtual void do_hltTriggerSelections(Strings& nameList) const;
      virtual void do_l1TriggerNames(Strings& nameList) const;

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

      uint32_t do_outputModuleId() const;
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
      mutable uint32_t _adler32;
      mutable uint32_t _outputModuleId;
      mutable std::string _outputModuleLabel;
      mutable Strings _hltTriggerNames;
      mutable Strings _hltTriggerSelections;
      mutable Strings _l1TriggerNames;

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

    private:

      void parseI2OHeader();
      void cacheHeaderFields() const;

      mutable bool _headerFieldsCached;
      mutable std::vector<unsigned char> _headerCopy;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable uint32_t _outputModuleId;
      mutable uint32_t _hltTriggerCount;
      mutable std::vector<unsigned char> _hltTriggerBits;
      mutable uint32_t _runNumber;
      mutable uint32_t _lumiSection;
      mutable uint32_t _eventNumber;
      mutable uint32_t _adler32;

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

      mutable bool _headerFieldsCached;
      mutable std::vector<unsigned char> _headerCopy;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable std::string _topFolderName;
      mutable DQMKey _dqmKey;
      mutable uint32_t _adler32;

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

      mutable bool _headerFieldsCached;
      mutable std::vector<unsigned char> _headerCopy;
      mutable unsigned long _headerSize;
      mutable unsigned char* _headerLocation;
      mutable uint32_t _runNumber;
      mutable uint32_t _lumiSection;
      mutable uint32_t _eventNumber;

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

      inline uint32_t do_runNumber() const { return _runNumber; }
      inline uint32_t do_lumiSection() const { return _lumiSection; }

    private:

      mutable uint32_t _runNumber;
      mutable uint32_t _lumiSection;

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

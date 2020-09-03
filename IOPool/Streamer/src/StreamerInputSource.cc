#include "IOPool/Streamer/interface/StreamerInputSource.h"

#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/ClassFiller.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"

#include "zlib.h"
#include "lzma.h"
#include "zstd.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <string>
#include <iostream>
#include <set>

namespace edm {
  namespace {
    int const init_size = 1024 * 1024;
  }

  StreamerInputSource::StreamerInputSource(ParameterSet const& pset, InputSourceDescription const& desc)
      : RawInputSource(pset, desc),
        tc_(getTClass(typeid(SendEvent))),
        dest_(init_size),
        xbuf_(TBuffer::kRead, init_size),
        sendEvent_(),
        eventPrincipalHolder_(),
        adjustEventToNewProductRegistry_(false),
        processName_(),
        protocolVersion_(0U) {}

  StreamerInputSource::~StreamerInputSource() {}

  // ---------------------------------------
  void StreamerInputSource::mergeIntoRegistry(SendJobHeader const& header,
                                              ProductRegistry& reg,
                                              BranchIDListHelper& branchIDListHelper,
                                              ThinnedAssociationsHelper& thinnedHelper,
                                              bool subsequent) {
    SendDescs const& descs = header.descs();

    FDEBUG(6) << "mergeIntoRegistry: Product List: " << std::endl;

    if (subsequent) {
      ProductRegistry pReg;
      pReg.updateFromInput(descs);
      std::string mergeInfo = reg.merge(pReg, std::string(), BranchDescription::Permissive);
      if (!mergeInfo.empty()) {
        throw cms::Exception("MismatchedInput", "RootInputFileSequence::previousEvent()") << mergeInfo;
      }
      branchIDListHelper.updateFromInput(header.branchIDLists());
      thinnedHelper.updateFromPrimaryInput(header.thinnedAssociationsHelper());
    } else {
      declareStreamers(descs);
      buildClassCache(descs);
      loadExtraClasses();
      if (!reg.frozen()) {
        reg.updateFromInput(descs);
      }
      branchIDListHelper.updateFromInput(header.branchIDLists());
      thinnedHelper.updateFromPrimaryInput(header.thinnedAssociationsHelper());
    }
  }

  void StreamerInputSource::declareStreamers(SendDescs const& descs) {
    std::vector<std::string> missingDictionaries;
    std::vector<std::string> branchNamesForMissing;
    std::vector<std::string> producedTypes;
    for (auto const& item : descs) {
      //pi->init();
      std::string const real_name = wrappedClassName(item.className());
      FDEBUG(6) << "declare: " << real_name << std::endl;
      if (!loadCap(real_name, missingDictionaries)) {
        branchNamesForMissing.emplace_back(item.branchName());
        producedTypes.emplace_back(item.className() + std::string(" (read from input)"));
      }
    }
    if (!missingDictionaries.empty()) {
      std::string context("Calling StreamerInputSource::declareStreamers, checking dictionaries for input types");
      throwMissingDictionariesException(missingDictionaries, context, producedTypes, branchNamesForMissing, true);
    }
  }

  void StreamerInputSource::buildClassCache(SendDescs const& descs) {
    for (auto const& item : descs) {
      //pi->init();
      std::string const real_name = wrappedClassName(item.className());
      FDEBUG(6) << "BuildReadData: " << real_name << std::endl;
      doBuildRealData(real_name);
    }
  }

  /**
   * Deserializes the specified init message into a SendJobHeader object
   * (which is related to the product registry).
   */
  std::unique_ptr<SendJobHeader> StreamerInputSource::deserializeRegistry(InitMsgView const& initView) {
    if (initView.code() != Header::INIT)
      throw cms::Exception("StreamTranslation", "Registry deserialization error")
          << "received wrong message type: expected INIT, got " << initView.code() << "\n";

    //Get the process name and store if for Protocol version 4 and above.
    if (initView.protocolVersion() > 3) {
      processName_ = initView.processName();
      protocolVersion_ = initView.protocolVersion();

      FDEBUG(10) << "StreamerInputSource::deserializeRegistry processName = " << processName_ << std::endl;
      FDEBUG(10) << "StreamerInputSource::deserializeRegistry protocolVersion_= " << protocolVersion_ << std::endl;
    }

    // calculate the adler32 checksum
    uint32_t adler32_chksum = cms::Adler32((char const*)initView.descData(), initView.descLength());
    //std::cout << "Adler32 checksum of init message = " << adler32_chksum << std::endl;
    //std::cout << "Adler32 checksum of init messsage from header = " << initView.adler32_chksum() << " "
    //          << "host name = " << initView.hostName() << " len = " << initView.hostName_len() << std::endl;
    if ((uint32)adler32_chksum != initView.adler32_chksum()) {
      // skip event (based on option?) or throw exception?
      throw cms::Exception("StreamDeserialization", "Checksum error")
          << " chksum from registry data = " << adler32_chksum << " from header = " << initView.adler32_chksum()
          << " host name = " << initView.hostName() << std::endl;
    }

    TClass* desc = getTClass(typeid(SendJobHeader));

    TBufferFile xbuf(
        TBuffer::kRead, initView.descLength(), const_cast<char*>((char const*)initView.descData()), kFALSE);
    RootDebug tracer(10, 10);
    std::unique_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

    if (sd.get() == nullptr) {
      throw cms::Exception("StreamTranslation", "Registry deserialization error")
          << "Could not read the initial product registry list\n";
    }

    sd->initializeTransients();
    return sd;
  }

  /**
   * Deserializes the specified init message into a SendJobHeader object
   * and merges registries.
   */
  void StreamerInputSource::deserializeAndMergeWithRegistry(InitMsgView const& initView, bool subsequent) {
    std::unique_ptr<SendJobHeader> sd = deserializeRegistry(initView);
    mergeIntoRegistry(*sd, productRegistryUpdate(), *branchIDListHelper(), *thinnedAssociationsHelper(), subsequent);
    if (subsequent) {
      adjustEventToNewProductRegistry_ = true;
    }
    SendJobHeader::ParameterSetMap const& psetMap = sd->processParameterSet();
    pset::Registry& psetRegistry = *pset::Registry::instance();
    for (auto const& item : psetMap) {
      ParameterSet pset(item.second.pset());
      pset.setID(item.first);
      psetRegistry.insertMapped(pset);
    }
  }

  /**
   * Deserializes the specified event message.
   */
  void StreamerInputSource::deserializeEvent(EventMsgView const& eventView) {
    if (eventView.code() != Header::EVENT)
      throw cms::Exception("StreamTranslation", "Event deserialization error")
          << "received wrong message type: expected EVENT, got " << eventView.code() << "\n";
    FDEBUG(9) << "Decode event: " << eventView.event() << " " << eventView.run() << " " << eventView.size() << " "
              << eventView.adler32_chksum() << " " << eventView.eventLength() << " " << eventView.eventData()
              << std::endl;
    // uncompress if we need to
    // 78 was a dummy value (for no uncompressed) - should be 0 for uncompressed
    // need to get rid of this when 090 MTCC streamers are gotten rid of
    unsigned long origsize = eventView.origDataSize();
    unsigned long dest_size;  //(should be >= eventView.origDataSize())

    uint32_t adler32_chksum = cms::Adler32((char const*)eventView.eventData(), eventView.eventLength());
    //std::cout << "Adler32 checksum of event = " << adler32_chksum << std::endl;
    //std::cout << "Adler32 checksum from header = " << eventView.adler32_chksum() << " "
    //          << "host name = " << eventView.hostName() << " len = " << eventView.hostName_len() << std::endl;
    if ((uint32)adler32_chksum != eventView.adler32_chksum()) {
      // skip event (based on option?) or throw exception?
      throw cms::Exception("StreamDeserialization", "Checksum error")
          << " chksum from event = " << adler32_chksum << " from header = " << eventView.adler32_chksum()
          << " host name = " << eventView.hostName() << std::endl;
    }
    if (origsize != 78 && origsize != 0) {
      // compressed
      if (isBufferLZMA((unsigned char const*)eventView.eventData(), eventView.eventLength())) {
        dest_size = uncompressBufferLZMA(const_cast<unsigned char*>((unsigned char const*)eventView.eventData()),
                                         eventView.eventLength(),
                                         dest_,
                                         origsize);
      } else if (isBufferZSTD((unsigned char const*)eventView.eventData(), eventView.eventLength())) {
        dest_size = uncompressBufferZSTD(const_cast<unsigned char*>((unsigned char const*)eventView.eventData()),
                                         eventView.eventLength(),
                                         dest_,
                                         origsize);
      } else
        dest_size = uncompressBuffer(const_cast<unsigned char*>((unsigned char const*)eventView.eventData()),
                                     eventView.eventLength(),
                                     dest_,
                                     origsize);
    } else {  // not compressed
      // we need to copy anyway the buffer as we are using dest in xbuf
      dest_size = eventView.eventLength();
      dest_.resize(dest_size);
      unsigned char* pos = (unsigned char*)&dest_[0];
      unsigned char const* from = (unsigned char const*)eventView.eventData();
      std::copy(from, from + dest_size, pos);
    }
    //TBuffer xbuf(TBuffer::kRead, dest_size,
    //             (char const*) &dest[0],kFALSE);
    //TBuffer xbuf(TBuffer::kRead, eventView.eventLength(),
    //             (char const*) eventView.eventData(),kFALSE);
    xbuf_.Reset();
    xbuf_.SetBuffer(&dest_[0], dest_size, kFALSE);
    RootDebug tracer(10, 10);

    //We do not yet know which EventPrincipal we will use, therefore
    // we are using a new EventPrincipalHolder as a proxy. We need to
    // make a new one instead of reusing the same one becuase when running
    // multi-threaded there will be multiple EventPrincipals being used
    // simultaneously.
    eventPrincipalHolder_ = std::make_unique<EventPrincipalHolder>();  // propagate_const<T> has no reset() function
    setRefCoreStreamer(eventPrincipalHolder_.get());
    {
      std::shared_ptr<void> refCoreStreamerGuard(nullptr, [](void*) {
        setRefCoreStreamer();
        ;
      });
      sendEvent_ = std::unique_ptr<SendEvent>((SendEvent*)xbuf_.ReadObjectAny(tc_));
    }

    if (sendEvent_.get() == nullptr) {
      throw cms::Exception("StreamTranslation", "Event deserialization error")
          << "got a null event from input stream\n";
    }
    processHistoryRegistryForUpdate().registerProcessHistory(sendEvent_->processHistory());

    FDEBUG(5) << "Got event: " << sendEvent_->aux().id() << " " << sendEvent_->products().size() << std::endl;
    if (runAuxiliary().get() == nullptr || runAuxiliary()->run() != sendEvent_->aux().run() ||
        runAuxiliary()->processHistoryID() != sendEvent_->processHistory().id()) {
      RunAuxiliary* runAuxiliary =
          new RunAuxiliary(sendEvent_->aux().run(), sendEvent_->aux().time(), Timestamp::invalidTimestamp());
      runAuxiliary->setProcessHistoryID(sendEvent_->processHistory().id());
      setRunAuxiliary(runAuxiliary);
      resetLuminosityBlockAuxiliary();
    }
    if (!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != eventView.lumi()) {
      LuminosityBlockAuxiliary* luminosityBlockAuxiliary = new LuminosityBlockAuxiliary(
          runAuxiliary()->run(), eventView.lumi(), sendEvent_->aux().time(), Timestamp::invalidTimestamp());
      luminosityBlockAuxiliary->setProcessHistoryID(sendEvent_->processHistory().id());
      setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
    }
    setEventCached();
  }

  void StreamerInputSource::read(EventPrincipal& eventPrincipal) {
    if (adjustEventToNewProductRegistry_) {
      eventPrincipal.adjustIndexesAfterProductRegistryAddition();
      bool eventOK = eventPrincipal.adjustToNewProductRegistry(*productRegistry());
      assert(eventOK);
      adjustEventToNewProductRegistry_ = false;
    }
    EventSelectionIDVector ids(sendEvent_->eventSelectionIDs());
    BranchListIndexes indexes(sendEvent_->branchListIndexes());
    branchIDListHelper()->fixBranchListIndexes(indexes);
    auto history = processHistoryRegistry().getMapped(sendEvent_->aux().processHistoryID());
    eventPrincipal.fillEventPrincipal(sendEvent_->aux(), history, std::move(ids), std::move(indexes));

    //We now know which eventPrincipal to use and we can reuse the slot in
    // streamToEventPrincipalHolders to own the memory
    eventPrincipalHolder_->setEventPrincipal(&eventPrincipal);
    if (streamToEventPrincipalHolders_.size() < eventPrincipal.streamID().value() + 1) {
      streamToEventPrincipalHolders_.resize(eventPrincipal.streamID().value() + 1);
    }
    streamToEventPrincipalHolders_[eventPrincipal.streamID().value()] = std::move(eventPrincipalHolder_);

    // no process name list handling

    SendProds& sps = sendEvent_->products();
    for (auto& spitem : sps) {
      FDEBUG(10) << "check prodpair" << std::endl;
      if (spitem.desc() == nullptr)
        throw cms::Exception("StreamTranslation", "Empty Provenance");
      FDEBUG(5) << "Prov:"
                << " " << spitem.desc()->className() << " " << spitem.desc()->productInstanceName() << " "
                << spitem.desc()->branchID() << std::endl;

      BranchDescription const branchDesc(*spitem.desc());
      // This ProductProvenance constructor inserts into the entry description registry
      if (spitem.parents()) {
        std::optional<ProductProvenance> productProvenance{std::in_place, spitem.branchID(), *spitem.parents()};
        if (spitem.prod() != nullptr) {
          FDEBUG(10) << "addproduct next " << spitem.branchID() << std::endl;
          eventPrincipal.putOnRead(branchDesc,
                                   std::unique_ptr<WrapperBase>(const_cast<WrapperBase*>(spitem.prod())),
                                   std::move(productProvenance));
          FDEBUG(10) << "addproduct done" << std::endl;
        } else {
          FDEBUG(10) << "addproduct empty next " << spitem.branchID() << std::endl;
          eventPrincipal.putOnRead(branchDesc, std::unique_ptr<WrapperBase>(), std::move(productProvenance));
          FDEBUG(10) << "addproduct empty done" << std::endl;
        }
      } else {
        std::optional<ProductProvenance> productProvenance;
        if (spitem.prod() != nullptr) {
          FDEBUG(10) << "addproduct next " << spitem.branchID() << std::endl;
          eventPrincipal.putOnRead(
              branchDesc, std::unique_ptr<WrapperBase>(const_cast<WrapperBase*>(spitem.prod())), productProvenance);
          FDEBUG(10) << "addproduct done" << std::endl;
        } else {
          FDEBUG(10) << "addproduct empty next " << spitem.branchID() << std::endl;
          eventPrincipal.putOnRead(branchDesc, std::unique_ptr<WrapperBase>(), productProvenance);
          FDEBUG(10) << "addproduct empty done" << std::endl;
        }
      }
      spitem.clear();
    }

    FDEBUG(10) << "Size = " << eventPrincipal.size() << std::endl;
  }

  /**
   * Uncompresses the data in the specified input buffer into the
   * specified output buffer.  The inputSize should be set to the size
   * of the compressed data in the inputBuffer.  The expectedFullSize should
   * be set to the original size of the data (before compression).
   * Returns the actual size of the uncompressed data.
   * Errors are reported by throwing exceptions.
   */
  unsigned int StreamerInputSource::uncompressBuffer(unsigned char* inputBuffer,
                                                     unsigned int inputSize,
                                                     std::vector<unsigned char>& outputBuffer,
                                                     unsigned int expectedFullSize) {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize * 1.1;
    FDEBUG(1) << "Uncompress: original size = " << origSize << ", compressed size = " << inputSize << std::endl;
    outputBuffer.resize(uncompressedSize);
    int ret = uncompress(&outputBuffer[0], &uncompressedSize, inputBuffer, inputSize);  // do not need compression level
    //std::cout << "unCompress Return value: " << ret << " Okay = " << Z_OK << std::endl;
    if (ret == Z_OK) {
      // check the length against original uncompressed length
      FDEBUG(10) << " original size = " << origSize << " final size = " << uncompressedSize << std::endl;
      if (origSize != uncompressedSize) {
        // we throw an error and return without event! null pointer
        throw cms::Exception("StreamDeserialization", "Uncompression error")
            << "mismatch event lengths should be" << origSize << " got " << uncompressedSize << "\n";
      }
    } else {
      // we throw an error and return without event! null pointer
      throw cms::Exception("StreamDeserialization", "Uncompression error") << "Error code = " << ret << "\n ";
    }
    return (unsigned int)uncompressedSize;
  }

  bool StreamerInputSource::isBufferLZMA(unsigned char const* inputBuffer, unsigned int inputSize) {
    if (inputSize >= 4 && !strcmp((const char*)inputBuffer, "XZ"))
      return true;
    else
      return false;
  }

  unsigned int StreamerInputSource::uncompressBufferLZMA(unsigned char* inputBuffer,
                                                         unsigned int inputSize,
                                                         std::vector<unsigned char>& outputBuffer,
                                                         unsigned int expectedFullSize,
                                                         bool hasHeader) {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize * 1.1;
    FDEBUG(1) << "Uncompress: original size = " << origSize << ", compressed size = " << inputSize << std::endl;
    outputBuffer.resize(uncompressedSize);

    lzma_stream stream = LZMA_STREAM_INIT;
    lzma_ret returnStatus;

    returnStatus = lzma_stream_decoder(&stream, UINT64_MAX, 0U);
    if (returnStatus != LZMA_OK) {
      throw cms::Exception("StreamDeserializationLZM", "LZMA stream decoder error")
          << "Error code = " << returnStatus << "\n ";
    }

    size_t hdrSize = hasHeader ? 4 : 0;
    stream.next_in = (const uint8_t*)(inputBuffer + hdrSize);
    stream.avail_in = (size_t)(inputSize - hdrSize);
    stream.next_out = (uint8_t*)&outputBuffer[0];
    stream.avail_out = (size_t)uncompressedSize;

    returnStatus = lzma_code(&stream, LZMA_FINISH);
    if (returnStatus != LZMA_STREAM_END) {
      lzma_end(&stream);
      throw cms::Exception("StreamDeserializationLZM", "LZMA uncompression error")
          << "Error code = " << returnStatus << "\n ";
    }
    lzma_end(&stream);

    uncompressedSize = (unsigned int)stream.total_out;

    FDEBUG(10) << " original size = " << origSize << " final size = " << uncompressedSize << std::endl;
    if (origSize != uncompressedSize) {
      // we throw an error and return without event! null pointer
      throw cms::Exception("StreamDeserialization", "LZMA uncompression error")
          << "mismatch event lengths should be" << origSize << " got " << uncompressedSize << "\n";
    }

    return uncompressedSize;
  }

  bool StreamerInputSource::isBufferZSTD(unsigned char const* inputBuffer, unsigned int inputSize) {
    if (inputSize >= 4 && !strcmp((const char*)inputBuffer, "ZS"))
      return true;
    else
      return false;
  }

  unsigned int StreamerInputSource::uncompressBufferZSTD(unsigned char* inputBuffer,
                                                         unsigned int inputSize,
                                                         std::vector<unsigned char>& outputBuffer,
                                                         unsigned int expectedFullSize,
                                                         bool hasHeader) {
    unsigned long uncompressedSize = expectedFullSize * 1.1;
    FDEBUG(1) << "Uncompress: original size = " << expectedFullSize << ", compressed size = " << inputSize << std::endl;
    outputBuffer.resize(uncompressedSize);

    size_t hdrSize = hasHeader ? 4 : 0;
    size_t ret = ZSTD_decompress(
        (void*)&(outputBuffer[0]), uncompressedSize, (const void*)(inputBuffer + hdrSize), inputSize - hdrSize);

    if (ZSTD_isError(ret)) {
      throw cms::Exception("StreamDeserializationZSTD", "ZSTD uncompression error")
          << "Error core " << ret << ", message:" << ZSTD_getErrorName(ret);
    }
    return (unsigned int)ret;
  }

  void StreamerInputSource::resetAfterEndRun() {
    // called from an online streamer source to reset after a stop command
    // so an enable command will work
    resetLuminosityBlockAuxiliary();
    resetRunAuxiliary();
    assert(!eventCached());
    reset();
  }

  void StreamerInputSource::setRun(RunNumber_t) {
    // Need to define a dummy setRun here or else the InputSource::setRun is called
    // if we have a source inheriting from this and wants to define a setRun method
    throw Exception(errors::LogicError) << "StreamerInputSource::setRun()\n"
                                        << "Run number cannot be modified for this type of Input Source\n"
                                        << "Contact a Storage Manager Developer\n";
  }

  StreamerInputSource::EventPrincipalHolder::EventPrincipalHolder() : eventPrincipal_(nullptr) {}

  StreamerInputSource::EventPrincipalHolder::~EventPrincipalHolder() {}

  WrapperBase const* StreamerInputSource::EventPrincipalHolder::getIt(ProductID const& id) const {
    return eventPrincipal_ ? eventPrincipal_->getIt(id) : nullptr;
  }

  std::optional<std::tuple<edm::WrapperBase const*, unsigned int>>
  StreamerInputSource::EventPrincipalHolder::getThinnedProduct(edm::ProductID const& id, unsigned int index) const {
    if (eventPrincipal_)
      return eventPrincipal_->getThinnedProduct(id, index);
    return std::nullopt;
  }

  void StreamerInputSource::EventPrincipalHolder::getThinnedProducts(ProductID const& pid,
                                                                     std::vector<WrapperBase const*>& wrappers,
                                                                     std::vector<unsigned int>& keys) const {
    if (eventPrincipal_)
      eventPrincipal_->getThinnedProducts(pid, wrappers, keys);
  }

  edm::OptionalThinnedKey StreamerInputSource::EventPrincipalHolder::getThinnedKeyFrom(
      edm::ProductID const& parent, unsigned int index, edm::ProductID const& thinned) const {
    if (eventPrincipal_) {
      return eventPrincipal_->getThinnedKeyFrom(parent, index, thinned);
    } else {
      return std::monostate{};
    }
  }

  unsigned int StreamerInputSource::EventPrincipalHolder::transitionIndex_() const {
    assert(eventPrincipal_ != nullptr);
    return eventPrincipal_->transitionIndex();
  }

  void StreamerInputSource::EventPrincipalHolder::setEventPrincipal(EventPrincipal* ep) { eventPrincipal_ = ep; }

  void StreamerInputSource::fillDescription(ParameterSetDescription& desc) { RawInputSource::fillDescription(desc); }
}  // namespace edm

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

#include "zlib.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/Utilities/interface/DebugMacros.h"

#include <string>
#include <iostream>
#include <set>

namespace edm {
  namespace {
    int const init_size = 1024*1024;
  }

  std::string StreamerInputSource::processName_;
  unsigned int StreamerInputSource::protocolVersion_;


  StreamerInputSource::StreamerInputSource(
                    ParameterSet const& pset,
                    InputSourceDescription const& desc):
    RawInputSource(pset, desc),
    tc_(getTClass(typeid(SendEvent))),
    dest_(init_size),
    xbuf_(TBuffer::kRead, init_size),
    sendEvent_(),
    productGetter_(),
    adjustEventToNewProductRegistry_(false) {
  }

  StreamerInputSource::~StreamerInputSource() {}

  // ---------------------------------------
  std::unique_ptr<FileBlock>
  StreamerInputSource::readFile_() {
    return std::unique_ptr<FileBlock>(new FileBlock);
  }

  void
  StreamerInputSource::mergeIntoRegistry(SendJobHeader const& header, ProductRegistry& reg, BranchIDListHelper& branchIDListHelper, bool subsequent) {

    SendDescs const& descs = header.descs();

    FDEBUG(6) << "mergeIntoRegistry: Product List: " << std::endl;

    if (subsequent) {
      ProductRegistry pReg;
      pReg.updateFromInput(descs);
      std::string mergeInfo = reg.merge(pReg, std::string(), BranchDescription::Permissive);
      if (!mergeInfo.empty()) {
        throw cms::Exception("MismatchedInput","RootInputFileSequence::previousEvent()") << mergeInfo;
      }
      branchIDListHelper.updateFromInput(header.branchIDLists());
    } else {
      declareStreamers(descs);
      buildClassCache(descs);
      loadExtraClasses();
      if(!reg.frozen()) {
        reg.updateFromInput(descs);
      }
      branchIDListHelper.updateFromInput(header.branchIDLists());
    }
  }

  void
  StreamerInputSource::declareStreamers(SendDescs const& descs) {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        std::string const real_name = wrappedClassName(i->className());
        FDEBUG(6) << "declare: " << real_name << std::endl;
        loadCap(real_name);
    }
  }


  void
  StreamerInputSource::buildClassCache(SendDescs const& descs) {
    SendDescs::const_iterator i(descs.begin()), e(descs.end());

    for(; i != e; ++i) {
        //pi->init();
        std::string const real_name = wrappedClassName(i->className());
        FDEBUG(6) << "BuildReadData: " << real_name << std::endl;
        doBuildRealData(real_name);
    }
  }

  /**
   * Deserializes the specified init message into a SendJobHeader object
   * (which is related to the product registry).
   */
  std::auto_ptr<SendJobHeader>
  StreamerInputSource::deserializeRegistry(InitMsgView const& initView) {
    if(initView.code() != Header::INIT)
      throw cms::Exception("StreamTranslation","Registry deserialization error")
        << "received wrong message type: expected INIT, got "
        << initView.code() << "\n";

    //Get the process name and store if for Protocol version 4 and above.
    if (initView.protocolVersion() > 3) {

         processName_ = initView.processName();
         protocolVersion_ = initView.protocolVersion();

         FDEBUG(10) << "StreamerInputSource::deserializeRegistry processName = "<< processName_<< std::endl;
         FDEBUG(10) << "StreamerInputSource::deserializeRegistry protocolVersion_= "<< protocolVersion_<< std::endl;
    }

   // calculate the adler32 checksum
   uint32_t adler32_chksum = cms::Adler32((char const*)initView.descData(),initView.descLength());
   //std::cout << "Adler32 checksum of init message = " << adler32_chksum << std::endl;
   //std::cout << "Adler32 checksum of init messsage from header = " << initView.adler32_chksum() << " "
   //          << "host name = " << initView.hostName() << " len = " << initView.hostName_len() << std::endl;
    if((uint32)adler32_chksum != initView.adler32_chksum()) {
      std::cerr << "Error from StreamerInputSource: checksum of Init registry blob failed "
                << " chksum from registry data = " << adler32_chksum << " from header = "
                << initView.adler32_chksum() << " host name = " << initView.hostName() << std::endl;
      // skip event (based on option?) or throw exception?
    }

    TClass* desc = getTClass(typeid(SendJobHeader));

    TBufferFile xbuf(TBuffer::kRead, initView.descLength(),
                 const_cast<char*>((char const*)initView.descData()),kFALSE);
    RootDebug tracer(10,10);
    std::auto_ptr<SendJobHeader> sd((SendJobHeader*)xbuf.ReadObjectAny(desc));

    if(sd.get()==0) {
        throw cms::Exception("StreamTranslation","Registry deserialization error")
          << "Could not read the initial product registry list\n";
    }

    sd->initializeTransients();
    return sd;
  }

  /**
   * Deserializes the specified init message into a SendJobHeader object
   * and merges registries.
   */
  void
  StreamerInputSource::deserializeAndMergeWithRegistry(InitMsgView const& initView, bool subsequent) {
     std::auto_ptr<SendJobHeader> sd = deserializeRegistry(initView);
     mergeIntoRegistry(*sd, productRegistryUpdate(), *branchIDListHelper(), subsequent);
     if (subsequent) {
       adjustEventToNewProductRegistry_ = true;
     }
     SendJobHeader::ParameterSetMap const& psetMap = sd->processParameterSet();
     pset::Registry& psetRegistry = *pset::Registry::instance();
     for (SendJobHeader::ParameterSetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end(); i != iEnd; ++i) {
       ParameterSet pset(i->second.pset());
       pset.setID(i->first);
       psetRegistry.insertMapped(pset);
     }
  }

  /**
   * Deserializes the specified event message.
   */
  void
  StreamerInputSource::deserializeEvent(EventMsgView const& eventView) {
    if(eventView.code() != Header::EVENT)
      throw cms::Exception("StreamTranslation","Event deserialization error")
        << "received wrong message type: expected EVENT, got "
        << eventView.code() << "\n";
    FDEBUG(9) << "Decode event: "
         << eventView.event() << " "
         << eventView.run() << " "
         << eventView.size() << " "
         << eventView.adler32_chksum() << " "
         << eventView.eventLength() << " "
         << eventView.eventData()
         << std::endl;
    EventSourceSentry sentry(*this);
    // uncompress if we need to
    // 78 was a dummy value (for no uncompressed) - should be 0 for uncompressed
    // need to get rid of this when 090 MTCC streamers are gotten rid of
    unsigned long origsize = eventView.origDataSize();
    unsigned long dest_size; //(should be >= eventView.origDataSize())

    uint32_t adler32_chksum = cms::Adler32((char const*)eventView.eventData(), eventView.eventLength());
    //std::cout << "Adler32 checksum of event = " << adler32_chksum << std::endl;
    //std::cout << "Adler32 checksum from header = " << eventView.adler32_chksum() << " "
    //          << "host name = " << eventView.hostName() << " len = " << eventView.hostName_len() << std::endl;
    if((uint32)adler32_chksum != eventView.adler32_chksum()) {
      std::cerr << "Error from StreamerInputSource: checksum of event data blob failed "
                << " chksum from event = " << adler32_chksum << " from header = "
                << eventView.adler32_chksum() << " host name = " << eventView.hostName() << std::endl;
      // skip event (based on option?) or throw exception?
    }
    if(origsize != 78 && origsize != 0) {
      // compressed
      dest_size = uncompressBuffer(const_cast<unsigned char*>((unsigned char const*)eventView.eventData()),
                                   eventView.eventLength(), dest_, origsize);
    } else { // not compressed
      // we need to copy anyway the buffer as we are using dest in xbuf
      dest_size = eventView.eventLength();
      dest_.resize(dest_size);
      unsigned char* pos = (unsigned char*) &dest_[0];
      unsigned char const* from = (unsigned char const*) eventView.eventData();
      std::copy(from,from+dest_size,pos);
    }
    //TBuffer xbuf(TBuffer::kRead, dest_size,
    //             (char const*) &dest[0],kFALSE);
    //TBuffer xbuf(TBuffer::kRead, eventView.eventLength(),
    //             (char const*) eventView.eventData(),kFALSE);
    xbuf_.Reset();
    xbuf_.SetBuffer(&dest_[0],dest_size,kFALSE);
    RootDebug tracer(10,10);

    setRefCoreStreamer(&productGetter_);
    sendEvent_ = std::unique_ptr<SendEvent>((SendEvent*)xbuf_.ReadObjectAny(tc_));
    setRefCoreStreamer();

    if(sendEvent_.get()==0) {
        throw cms::Exception("StreamTranslation","Event deserialization error")
          << "got a null event from input stream\n";
    }
    ProcessHistoryRegistry::instance()->insertMapped(sendEvent_->processHistory());

    FDEBUG(5) << "Got event: " << sendEvent_->aux().id() << " " << sendEvent_->products().size() << std::endl;
    if(runAuxiliary().get() == 0 || runAuxiliary()->run() != sendEvent_->aux().run()) {
      RunAuxiliary* runAuxiliary = new RunAuxiliary(sendEvent_->aux().run(), sendEvent_->aux().time(), Timestamp::invalidTimestamp());
      runAuxiliary->setProcessHistoryID(sendEvent_->processHistory().id());
      setRunAuxiliary(runAuxiliary);
      resetLuminosityBlockAuxiliary();
    }
    if(!luminosityBlockAuxiliary() || luminosityBlockAuxiliary()->luminosityBlock() != eventView.lumi()) {
      LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
        new LuminosityBlockAuxiliary(runAuxiliary()->run(), eventView.lumi(), sendEvent_->aux().time(), Timestamp::invalidTimestamp());
      luminosityBlockAuxiliary->setProcessHistoryID(sendEvent_->processHistory().id());
      setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
    }
    setEventCached();
  }

  EventPrincipal *
  StreamerInputSource::read(EventPrincipal& eventPrincipal) {
    if(adjustEventToNewProductRegistry_) {
      eventPrincipal.adjustIndexesAfterProductRegistryAddition();
      bool eventOK = eventPrincipal.adjustToNewProductRegistry(*productRegistry());
      assert(eventOK);
      adjustEventToNewProductRegistry_ = false;
    }
    boost::shared_ptr<EventSelectionIDVector> ids(new EventSelectionIDVector(sendEvent_->eventSelectionIDs()));
    boost::shared_ptr<BranchListIndexes> indexes(new BranchListIndexes(sendEvent_->branchListIndexes()));
    branchIDListHelper()->fixBranchListIndexes(*indexes);
    eventPrincipal.fillEventPrincipal(sendEvent_->aux(), ids, indexes);
    productGetter_.setEventPrincipal(&eventPrincipal);

    // no process name list handling

    SendProds & sps = sendEvent_->products();
    for(SendProds::iterator spi = sps.begin(), spe = sps.end(); spi != spe; ++spi) {
        FDEBUG(10) << "check prodpair" << std::endl;
        if(spi->desc() == 0)
          throw cms::Exception("StreamTranslation","Empty Provenance");
        FDEBUG(5) << "Prov:"
             << " " << spi->desc()->className()
             << " " << spi->desc()->productInstanceName()
             << " " << spi->desc()->branchID()
             << std::endl;

        BranchDescription const branchDesc(*spi->desc());
        // This ProductProvenance constructor inserts into the entry description registry
        ProductProvenance productProvenance(spi->branchID(), *spi->parents());

        if(spi->prod() != 0) {
          FDEBUG(10) << "addproduct next " << spi->branchID() << std::endl;
          eventPrincipal.putOnRead(branchDesc, spi->prod(), productProvenance);
          FDEBUG(10) << "addproduct done" << std::endl;
        } else {
          FDEBUG(10) << "addproduct empty next " << spi->branchID() << std::endl;
          eventPrincipal.putOnRead(branchDesc, spi->prod(), productProvenance);
          FDEBUG(10) << "addproduct empty done" << std::endl;
        }
        spi->clear();
    }

    FDEBUG(10) << "Size = " << eventPrincipal.size() << std::endl;

    return &eventPrincipal;
  }

  /**
   * Uncompresses the data in the specified input buffer into the
   * specified output buffer.  The inputSize should be set to the size
   * of the compressed data in the inputBuffer.  The expectedFullSize should
   * be set to the original size of the data (before compression).
   * Returns the actual size of the uncompressed data.
   * Errors are reported by throwing exceptions.
   */
  unsigned int
  StreamerInputSource::uncompressBuffer(unsigned char* inputBuffer,
                                        unsigned int inputSize,
                                        std::vector<unsigned char>& outputBuffer,
                                        unsigned int expectedFullSize) {
    unsigned long origSize = expectedFullSize;
    unsigned long uncompressedSize = expectedFullSize*1.1;
    FDEBUG(1) << "Uncompress: original size = " << origSize
              << ", compressed size = " << inputSize
              << std::endl;
    outputBuffer.resize(uncompressedSize);
    int ret = uncompress(&outputBuffer[0], &uncompressedSize,
                         inputBuffer, inputSize); // do not need compression level
    //std::cout << "unCompress Return value: " << ret << " Okay = " << Z_OK << std::endl;
    if(ret == Z_OK) {
        // check the length against original uncompressed length
        FDEBUG(10) << " original size = " << origSize << " final size = "
                   << uncompressedSize << std::endl;
        if(origSize != uncompressedSize) {
            std::cerr << "deserializeEvent: Problem with uncompress, original size = "
                 << origSize << " uncompress size = " << uncompressedSize << std::endl;
            // we throw an error and return without event! null pointer
            throw cms::Exception("StreamDeserialization","Uncompression error")
              << "mismatch event lengths should be" << origSize << " got "
              << uncompressedSize << "\n";
        }
    } else {
        // we throw an error and return without event! null pointer
        std::cerr << "deserializeEvent: Problem with uncompress, return value = "
             << ret << std::endl;
        throw cms::Exception("StreamDeserialization","Uncompression error")
            << "Error code = " << ret << "\n ";
    }
    return (unsigned int) uncompressedSize;
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
     throw Exception(errors::LogicError)
     << "StreamerInputSource::setRun()\n"
     << "Run number cannot be modified for this type of Input Source\n"
     << "Contact a Storage Manager Developer\n";
  }

  StreamerInputSource::ProductGetter::ProductGetter() : eventPrincipal_(0) {}

  StreamerInputSource::ProductGetter::~ProductGetter() {}

  WrapperHolder
  StreamerInputSource::ProductGetter::getIt(ProductID const& id) const {
    return eventPrincipal_ ? eventPrincipal_->getIt(id) : WrapperHolder();
  }

  void
  StreamerInputSource::ProductGetter::setEventPrincipal(EventPrincipal *ep) {
    eventPrincipal_ = ep;
  }

  void
  StreamerInputSource::fillDescription(ParameterSetDescription& desc) {
    RawInputSource::fillDescription(desc);
  }
}

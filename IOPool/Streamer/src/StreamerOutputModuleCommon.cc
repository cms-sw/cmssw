#include "IOPool/Streamer/interface/StreamerOutputModuleCommon.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/SelectedProducts.h"
#include "FWCore/Framework/interface/getAllTriggerNames.h"

#include <iostream>
#include <memory>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <zlib.h>

namespace edm {
  StreamerOutputModuleCommon::Parameters StreamerOutputModuleCommon::parameters(ParameterSet const& ps) {
    Parameters ret;
    ret.hltTriggerSelections = EventSelector::getEventSelectionVString(ps);
    ret.compressionAlgoStr = ps.getUntrackedParameter<std::string>("compression_algorithm");
    ret.compressionLevel = ps.getUntrackedParameter<int>("compression_level");
    ret.lumiSectionInterval = ps.getUntrackedParameter<int>("lumiSection_interval");
    ret.useCompression = ps.getUntrackedParameter<bool>("use_compression");
    return ret;
  }

  StreamerOutputModuleCommon::StreamerOutputModuleCommon(Parameters const& p,
                                                         SelectedProducts const* selections,
                                                         std::string const& moduleLabel)
      :

        serializer_(selections),
        useCompression_(p.useCompression),
        compressionAlgoStr_(p.compressionAlgoStr),
        compressionLevel_(p.compressionLevel),
        lumiSectionInterval_(p.lumiSectionInterval),
        hltsize_(0),
        host_name_(),
        hltTriggerSelections_(),
        outputModuleId_(0) {
    //limits initially set for default ZLIB option
    int minCompressionLevel = 1;
    int maxCompressionLevel = 9;

    // test luminosity sections
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    timeInSecSinceUTC = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec) / 1000000.0);

    if (useCompression_ == true) {
      if (compressionAlgoStr_ == "ZLIB") {
        compressionAlgo_ = ZLIB;
      } else if (compressionAlgoStr_ == "LZMA") {
        compressionAlgo_ = LZMA;
        minCompressionLevel = 0;
      } else if (compressionAlgoStr_ == "ZSTD") {
        compressionAlgo_ = ZSTD;
        maxCompressionLevel = 20;
      } else if (compressionAlgoStr_ == "UNCOMPRESSED") {
        compressionLevel_ = 0;
        useCompression_ = false;
        compressionAlgo_ = UNCOMPRESSED;
      } else
        throw cms::Exception("StreamerOutputModuleCommon", "Compression type unknown")
            << "Unknown compression algorithm " << compressionAlgoStr_;

      if (compressionLevel_ < minCompressionLevel) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_ << " no compression" << std::endl;
        compressionLevel_ = 0;
        useCompression_ = false;
        compressionAlgo_ = UNCOMPRESSED;
      } else if (compressionLevel_ > maxCompressionLevel) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_ << " using max compression level "
                  << maxCompressionLevel << std::endl;
        compressionLevel_ = maxCompressionLevel;
        compressionAlgo_ = UNCOMPRESSED;
      }
    } else
      compressionAlgo_ = UNCOMPRESSED;

    int got_host = gethostname(host_name_, 255);
    if (got_host != 0)
      strncpy(host_name_, "noHostNameFoundOrTooLong", sizeof(host_name_));
    //loadExtraClasses();

    // 25-Jan-2008, KAB - pull out the trigger selection request
    // which we need for the INIT message
    hltTriggerSelections_ = p.hltTriggerSelections;

    Strings const& hltTriggerNames = edm::getAllTriggerNames();
    hltsize_ = hltTriggerNames.size();

    //Checksum of the module label
    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef const* buf = (Bytef const*)(moduleLabel.data());
    crc = crc32(crc, buf, moduleLabel.length());
    outputModuleId_ = static_cast<uint32>(crc);
  }

  StreamerOutputModuleCommon::~StreamerOutputModuleCommon() {}

  std::unique_ptr<InitMsgBuilder> StreamerOutputModuleCommon::serializeRegistry(
      SerializeDataBuffer& sbuf,
      const BranchIDLists& branchLists,
      ThinnedAssociationsHelper const& helper,
      std::string const& processName,
      std::string const& moduleLabel,
      ParameterSetID const& toplevel,
      SendJobHeader::ParameterSetMap const* psetMap) {
    if (psetMap) {
      serializer_.serializeRegistry(sbuf, branchLists, helper, *psetMap);
    } else {
      serializer_.serializeRegistry(sbuf, branchLists, helper);
    }
    // resize header_buf_ to reflect space used in serializer_ + header
    // I just added an overhead for header of 50000 for now
    unsigned int src_size = sbuf.currentSpaceUsed();
    unsigned int new_size = src_size + 50000;
    if (sbuf.header_buf_.size() < new_size)
      sbuf.header_buf_.resize(new_size);

    //Build the INIT Message
    //Following values are strictly DUMMY and will be replaced
    // once available with Utility function etc.
    uint32 run = 1;

    //Get the Process PSet ID

    //In case we need to print it
    //  cms::Digest dig(toplevel.compactForm());
    //  cms::MD5Result r1 = dig.digest();
    //  std::string hexy = r1.toString();
    //  std::cout << "HEX Representation of Process PSetID: " << hexy << std::endl;

    //L1 stays dummy as of today
    Strings l1_names;  //3
    l1_names.push_back("t1");
    l1_names.push_back("t10");
    l1_names.push_back("t2");

    Strings const& hltTriggerNames = edm::getAllTriggerNames();

    auto init_message = std::make_unique<InitMsgBuilder>(&sbuf.header_buf_[0],
                                                         sbuf.header_buf_.size(),
                                                         run,
                                                         Version((uint8 const*)toplevel.compactForm().c_str()),
                                                         getReleaseVersion().c_str(),
                                                         processName.c_str(),
                                                         moduleLabel.c_str(),
                                                         outputModuleId_,
                                                         hltTriggerNames,
                                                         hltTriggerSelections_,
                                                         l1_names,
                                                         (uint32)sbuf.adler32_chksum());

    // copy data into the destination message
    unsigned char* src = sbuf.bufferPointer();
    std::copy(src, src + src_size, init_message->dataAddress());
    init_message->setDataLength(src_size);
    return init_message;
  }

  void StreamerOutputModuleCommon::setHltMask(EventForOutput const& e,
                                              Handle<TriggerResults> const& triggerResults,
                                              std::vector<unsigned char>& hltbits) const {
    hltbits.clear();

    std::vector<unsigned char> vHltState;

    if (triggerResults.isValid()) {
      for (std::vector<unsigned char>::size_type i = 0; i != hltsize_; ++i) {
        vHltState.push_back(((triggerResults->at(i)).state()));
      }
    } else {
      // We fill all Trigger bits to valid state.
      for (std::vector<unsigned char>::size_type i = 0; i != hltsize_; ++i) {
        vHltState.push_back(hlt::Pass);
      }
    }

    //Pack into member hltbits
    if (!vHltState.empty()) {
      unsigned int packInOneByte = 4;
      unsigned int sizeOfPackage = 1 + ((vHltState.size() - 1) / packInOneByte);  //Two bits per HLT

      hltbits.resize(sizeOfPackage);
      std::fill(hltbits.begin(), hltbits.end(), 0);

      for (std::vector<unsigned char>::size_type i = 0; i != vHltState.size(); ++i) {
        unsigned int whichByte = i / packInOneByte;
        unsigned int indxWithinByte = i % packInOneByte;
        hltbits[whichByte] = hltbits[whichByte] | (vHltState[i] << (indxWithinByte * 2));
      }
    }

    //This is Just a printing code.
    //std::cout << "Size of hltbits:" << hltbits_.size() << std::endl;
    //for(unsigned int i=0; i != hltbits_.size() ; ++i) {
    //  printBits(hltbits_[i]);
    //}
    //std::cout << "\n";
  }

  std::unique_ptr<EventMsgBuilder> StreamerOutputModuleCommon::serializeEvent(
      SerializeDataBuffer& sbuf,
      EventForOutput const& e,
      Handle<TriggerResults> const& triggerResults,
      ParameterSetID const& selectorCfg) {
    constexpr unsigned int reserve_size = SerializeDataBuffer::reserve_size;
    //Lets Build the Event Message first

    //Following is strictly DUMMY Data for L! Trig and will be replaced with actual
    // once figured out, there is no logic involved here.
    std::vector<bool> l1bit = {true, true, false};
    //End of dummy data

    std::vector<unsigned char> hltbits;
    setHltMask(e, triggerResults, hltbits);

    uint32 lumi;
    if (lumiSectionInterval_ == 0) {
      lumi = e.luminosityBlock();
    } else {
      struct timeval now;
      struct timezone dummyTZ;
      gettimeofday(&now, &dummyTZ);
      double timeInSec =
          static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec) / 1000000.0) - timeInSecSinceUTC;
      // what about overflows?
      if (lumiSectionInterval_ > 0)
        lumi = static_cast<uint32>(timeInSec / lumiSectionInterval_) + 1;
    }

    serializer_.serializeEvent(sbuf, e, selectorCfg, compressionAlgo_, compressionLevel_, reserve_size);

    // resize header_buf_ to reserved size on first written event
    if (sbuf.header_buf_.size() < reserve_size)
      sbuf.header_buf_.resize(reserve_size);

    auto msg = std::make_unique<EventMsgBuilder>(&sbuf.header_buf_[0],
                                                 sbuf.comp_buf_.size(),
                                                 e.id().run(),
                                                 e.id().event(),
                                                 lumi,
                                                 outputModuleId_,
                                                 0,
                                                 l1bit,
                                                 (uint8*)&hltbits[0],
                                                 hltsize_,
                                                 (uint32)sbuf.adler32_chksum(),
                                                 host_name_);

    // 50000 bytes is reserved for header as has been the case with previous version which did one extra copy of event data
    uint32 headerSize = msg->headerSize();
    if (headerSize > reserve_size)
      throw cms::Exception("StreamerOutputModuleCommon", "Header Overflow")
          << " header of size " << headerSize << "bytes is too big to fit into the reserved buffer space";

    //set addresses to other buffer and copy constructed header there
    msg->setBufAddr(&sbuf.comp_buf_[reserve_size - headerSize]);
    msg->setEventAddr(sbuf.bufferPointer());
    std::copy(&sbuf.header_buf_[0], &sbuf.header_buf_[headerSize], (char*)(&sbuf.comp_buf_[reserve_size - headerSize]));

    unsigned int src_size = sbuf.currentSpaceUsed();
    msg->setEventLength(src_size);  //compressed size
    if (useCompression_)
      msg->setOrigDataSize(
          sbuf.currentEventSize());  //uncompressed size (or 0 if no compression -> streamer input source requires this)
    else
      msg->setOrigDataSize(0);

    return msg;
  }

  void StreamerOutputModuleCommon::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<int>("max_event_size", 7000000)->setComment("Obsolete parameter.");
    desc.addUntracked<bool>("use_compression", true)
        ->setComment("If True, compression will be used to write streamer file.");
    desc.addUntracked<std::string>("compression_algorithm", "ZLIB")
        ->setComment("Compression algorithm to use: UNCOMPRESSED, ZLIB, LZMA or ZSTD");
    desc.addUntracked<int>("compression_level", 1)->setComment("Compression level to use on serialized ROOT events");
    desc.addUntracked<int>("lumiSection_interval", 0)
        ->setComment(
            "If 0, use lumi section number from event.\n"
            "If not 0, the interval in seconds between fake lumi sections.");
  }

  SerializeDataBuffer* StreamerOutputModuleCommon::getSerializerBuffer() {
    auto* ptr = serializerBuffer_.get();
    if (!ptr) {
      serializerBuffer_ = std::make_unique<SerializeDataBuffer>();
      ptr = serializerBuffer_.get();
    }
    return ptr;
  }
}  // namespace edm

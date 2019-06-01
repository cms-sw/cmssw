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
  StreamerOutputModuleCommon::StreamerOutputModuleCommon(ParameterSet const& ps, SelectedProducts const* selections)
      :

        serializer_(selections),
        maxEventSize_(ps.getUntrackedParameter<int>("max_event_size")),
        useCompression_(ps.getUntrackedParameter<bool>("use_compression")),
        compressionLevel_(ps.getUntrackedParameter<int>("compression_level")),
        lumiSectionInterval_(ps.getUntrackedParameter<int>("lumiSection_interval")),
        serializeDataBuffer_(),
        hltsize_(0),
        origSize_(0),
        host_name_(),
        hltTriggerSelections_(),
        outputModuleId_(0) {
    // no compression as default value - we need this!

    // test luminosity sections
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    timeInSecSinceUTC = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec) / 1000000.0);

    if (useCompression_ == true) {
      if (compressionLevel_ <= 0) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_ << " no compression" << std::endl;
        compressionLevel_ = 0;
        useCompression_ = false;
      } else if (compressionLevel_ > 9) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_ << " using max compression level 9" << std::endl;
        compressionLevel_ = 9;
      }
    }
    serializeDataBuffer_.bufs_.resize(maxEventSize_);
    int got_host = gethostname(host_name_, 255);
    if (got_host != 0)
      strncpy(host_name_, "noHostNameFoundOrTooLong", sizeof(host_name_));
    //loadExtraClasses();

    // 25-Jan-2008, KAB - pull out the trigger selection request
    // which we need for the INIT message
    hltTriggerSelections_ = EventSelector::getEventSelectionVString(ps);
  }

  StreamerOutputModuleCommon::~StreamerOutputModuleCommon() {}

  std::unique_ptr<InitMsgBuilder> StreamerOutputModuleCommon::serializeRegistry(const BranchIDLists& branchLists,
                                                                                ThinnedAssociationsHelper const& helper,
                                                                                std::string const& processName,
                                                                                std::string const& moduleLabel,
                                                                                ParameterSetID const& toplevel) {
    serializer_.serializeRegistry(serializeDataBuffer_, branchLists, helper);

    // resize bufs_ to reflect space used in serializer_ + header
    // I just added an overhead for header of 50000 for now
    unsigned int src_size = serializeDataBuffer_.currentSpaceUsed();
    unsigned int new_size = src_size + 50000;
    if (serializeDataBuffer_.header_buf_.size() < new_size)
      serializeDataBuffer_.header_buf_.resize(new_size);

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
    Strings const& hltTriggerNames = edm::getAllTriggerNames();
    hltsize_ = hltTriggerNames.size();

    //L1 stays dummy as of today
    Strings l1_names;  //3
    l1_names.push_back("t1");
    l1_names.push_back("t10");
    l1_names.push_back("t2");

    //Setting the process name to HLT
    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef const* buf = (Bytef const*)(moduleLabel.data());
    crc = crc32(crc, buf, moduleLabel.length());
    outputModuleId_ = static_cast<uint32>(crc);

    auto init_message = std::make_unique<InitMsgBuilder>(&serializeDataBuffer_.header_buf_[0],
                                                         serializeDataBuffer_.header_buf_.size(),
                                                         run,
                                                         Version((uint8 const*)toplevel.compactForm().c_str()),
                                                         getReleaseVersion().c_str(),
                                                         processName.c_str(),
                                                         moduleLabel.c_str(),
                                                         outputModuleId_,
                                                         hltTriggerNames,
                                                         hltTriggerSelections_,
                                                         l1_names,
                                                         (uint32)serializeDataBuffer_.adler32_chksum());

    // copy data into the destination message
    unsigned char* src = serializeDataBuffer_.bufferPointer();
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
      EventForOutput const& e, Handle<TriggerResults> const& triggerResults, ParameterSetID const& selectorCfg) {
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

    serializer_.serializeEvent(e, selectorCfg, useCompression_, compressionLevel_, serializeDataBuffer_);

    // resize bufs_ to reflect space used in serializer_ + header
    // I just added an overhead for header of 50000 for now
    unsigned int src_size = serializeDataBuffer_.currentSpaceUsed();
    unsigned int new_size = src_size + 50000;
    if (serializeDataBuffer_.bufs_.size() < new_size)
      serializeDataBuffer_.bufs_.resize(new_size);

    auto msg = std::make_unique<EventMsgBuilder>(&serializeDataBuffer_.bufs_[0],
                                                 serializeDataBuffer_.bufs_.size(),
                                                 e.id().run(),
                                                 e.id().event(),
                                                 lumi,
                                                 outputModuleId_,
                                                 0,
                                                 l1bit,
                                                 (uint8*)&hltbits[0],
                                                 hltsize_,
                                                 (uint32)serializeDataBuffer_.adler32_chksum(),
                                                 host_name_);
    msg->setOrigDataSize(origSize_);  // we need this set to zero

    // copy data into the destination message
    // an alternative is to have serializer only to the serialization
    // in serializeEvent, and then call a new member "getEventData" that
    // takes the compression arguments and a place to put the data.
    // This will require one less copy.  The only catch is that the
    // space provided in bufs_ should be at least the uncompressed
    // size + overhead for header because we will not know the actual
    // compressed size.

    unsigned char* src = serializeDataBuffer_.bufferPointer();
    std::copy(src, src + src_size, msg->eventAddr());
    msg->setEventLength(src_size);
    if (useCompression_)
      msg->setOrigDataSize(serializeDataBuffer_.currentEventSize());

    return msg;
  }

  void StreamerOutputModuleCommon::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<int>("max_event_size", 7000000)
        ->setComment("Starting size in bytes of the serialized event buffer.");
    desc.addUntracked<bool>("use_compression", true)
        ->setComment("If True, compression will be used to write streamer file.");
    desc.addUntracked<int>("compression_level", 1)->setComment("ROOT compression level to use.");
    desc.addUntracked<int>("lumiSection_interval", 0)
        ->setComment(
            "If 0, use lumi section number from event.\n"
            "If not 0, the interval in seconds between fake lumi sections.");
  }
}  // namespace edm

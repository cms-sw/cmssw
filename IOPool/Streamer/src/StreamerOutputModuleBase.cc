
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
//#include "FWCore/Utilities/interface/Digest.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

#include <string>
#include <unistd.h>
#include "zlib.h"

static SerializeDataBuffer serialize_databuffer;

namespace {
  //A utility function that packs bits from source into bytes, with
  // packInOneByte as the numeber of bytes that are packed from source to dest.
  void printBits(unsigned char c) {
    for (int i = 7; i >= 0; --i) {
      int bit = ((c >> i) & 1);
      std::cout << " " << bit;
    }
  }

  void packIntoString(std::vector<unsigned char> const& source,
                      std::vector<unsigned char>& package) {
     if (source.size() < 1) {return;}
     unsigned int packInOneByte = 4;
     unsigned int sizeOfPackage = 1+((source.size()-1)/packInOneByte); //Two bits per HLT

     package.resize(sizeOfPackage);
     memset(&package[0], 0x00, sizeOfPackage);

     for (std::vector<unsigned char>::size_type i=0; i != source.size() ; ++i) {
       unsigned int whichByte = i/packInOneByte;
       unsigned int indxWithinByte = i % packInOneByte;
       package[whichByte] = package[whichByte] | (source[i] << (indxWithinByte*2));
     }
    //for (unsigned int i=0; i !=package.size() ; ++i)
    //   printBits(package[i]);
    // std::cout << std::endl;

  }
}

namespace edm {
  StreamerOutputModuleBase::StreamerOutputModuleBase(ParameterSet const& ps) :
    OutputModule(ps),
    selections_(&keptProducts()[InEvent]),
    maxEventSize_(ps.getUntrackedParameter<int>("max_event_size")),
    useCompression_(ps.getUntrackedParameter<bool>("use_compression")),
    compressionLevel_(ps.getUntrackedParameter<int>("compression_level")),
    lumiSectionInterval_(ps.getUntrackedParameter<int>("lumiSection_interval")),
    serializer_(selections_),
    hltsize_(0),
    lumi_(0),
    l1bit_(0),
    hltbits_(0),
    origSize_(0),
    host_name_(),
    hltTriggerSelections_(),
    outputModuleId_(0) {
    // no compression as default value - we need this!

    // test luminosity sections
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    timeInSecSinceUTC = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec)/1000000.0);

    if(useCompression_ == true) {
      if(compressionLevel_ <= 0) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_
                  << " no compression" << std::endl;
        compressionLevel_ = 0;
        useCompression_ = false;
      } else if(compressionLevel_ > 9) {
        FDEBUG(9) << "Compression Level = " << compressionLevel_
                  << " using max compression level 9" << std::endl;
        compressionLevel_ = 9;
      }
    }
    serialize_databuffer.bufs_.resize(maxEventSize_);
    int got_host = gethostname(host_name_, 255);
    if(got_host != 0) strncpy(host_name_, "noHostNameFoundOrTooLong", sizeof(host_name_));
    //loadExtraClasses();
    // do the line below instead of loadExtraClasses() to avoid Root errors
    RootAutoLibraryLoader::enable();

    // 25-Jan-2008, KAB - pull out the trigger selection request
    // which we need for the INIT message
    hltTriggerSelections_ = EventSelector::getEventSelectionVString(ps);
  }

  StreamerOutputModuleBase::~StreamerOutputModuleBase() {}

  void
  StreamerOutputModuleBase::beginRun(RunPrincipal const&, ModuleCallingContext const*) {
    start();
    std::auto_ptr<InitMsgBuilder>  init_message = serializeRegistry();
    doOutputHeader(*init_message);
  }

  void
  StreamerOutputModuleBase::endRun(RunPrincipal const&, ModuleCallingContext const*) {
    stop();
  }

  void
  StreamerOutputModuleBase::beginJob() {}

  void
  StreamerOutputModuleBase::endJob() {
    stop();  // for closing of files, notify storage manager, etc.
  }

  void
  StreamerOutputModuleBase::writeRun(RunPrincipal const&, ModuleCallingContext const*) {}

  void
  StreamerOutputModuleBase::writeLuminosityBlock(LuminosityBlockPrincipal const&, ModuleCallingContext const*) {}

  void
  StreamerOutputModuleBase::write(EventPrincipal const& e, ModuleCallingContext const* mcc) {
    std::auto_ptr<EventMsgBuilder> msg = serializeEvent(e, mcc);
    doOutputEvent(*msg); // You can't use msg in StreamerOutputModuleBase after this point
  }

  std::auto_ptr<InitMsgBuilder>
  StreamerOutputModuleBase::serializeRegistry() {

    serializer_.serializeRegistry(serialize_databuffer, *branchIDLists());

    // resize bufs_ to reflect space used in serializer_ + header
    // I just added an overhead for header of 50000 for now
    unsigned int src_size = serialize_databuffer.currentSpaceUsed();
    unsigned int new_size = src_size + 50000;
    if(serialize_databuffer.header_buf_.size() < new_size) serialize_databuffer.header_buf_.resize(new_size);

    //Build the INIT Message
    //Following values are strictly DUMMY and will be replaced
    // once available with Utility function etc.
    uint32 run = 1;

    //Get the Process PSet ID
    pset::Registry const& reg = *pset::Registry::instance();
    ParameterSetID toplevel = pset::getProcessParameterSetID(reg);

    //In case we need to print it
    //  cms::Digest dig(toplevel.compactForm());
    //  cms::MD5Result r1 = dig.digest();
    //  std::string hexy = r1.toString();
    //  std::cout << "HEX Representation of Process PSetID: " << hexy << std::endl;

    Strings hltTriggerNames = getAllTriggerNames();
    hltsize_ = hltTriggerNames.size();

    //L1 stays dummy as of today
    Strings l1_names;  //3
    l1_names.push_back("t1");
    l1_names.push_back("t10");
    l1_names.push_back("t2");

    //Setting the process name to HLT
    std::string processName = OutputModule::processName();

    std::string moduleLabel = description().moduleLabel();
    uLong crc = crc32(0L, Z_NULL, 0);
    Bytef* buf = (Bytef*) moduleLabel.data();
    crc = crc32(crc, buf, moduleLabel.length());
    outputModuleId_ = static_cast<uint32>(crc);

    std::auto_ptr<InitMsgBuilder> init_message(
        new InitMsgBuilder(&serialize_databuffer.header_buf_[0], serialize_databuffer.header_buf_.size(),
                           run, Version((uint8*)toplevel.compactForm().c_str()),
                           getReleaseVersion().c_str() , processName.c_str(),
                           moduleLabel.c_str(), outputModuleId_,
                           hltTriggerNames, hltTriggerSelections_, l1_names,
                           (uint32)serialize_databuffer.adler32_chksum(), host_name_));

    // copy data into the destination message
    unsigned char* src = serialize_databuffer.bufferPointer();
    std::copy(src, src + src_size, init_message->dataAddress());
    init_message->setDataLength(src_size);
    return init_message;
  }

  void
  StreamerOutputModuleBase::setHltMask(EventPrincipal const& e, ModuleCallingContext const* mcc) {

    hltbits_.clear();  // If there was something left over from last event

    Handle<TriggerResults> const& prod = getTriggerResults(e, mcc);
    //Trig const& prod = getTrigMask(e);
    std::vector<unsigned char> vHltState;

    if (prod.isValid()) {
      for(std::vector<unsigned char>::size_type i=0; i != hltsize_ ; ++i) {
        vHltState.push_back(((prod->at(i)).state()));
      }
    } else {
     // We fill all Trigger bits to valid state.
     for(std::vector<unsigned char>::size_type i=0; i != hltsize_ ; ++i) {
           vHltState.push_back(hlt::Pass);
      }
    }
    //Pack into member hltbits_
    packIntoString(vHltState, hltbits_);

    //This is Just a printing code.
    //std::cout << "Size of hltbits:" << hltbits_.size() << std::endl;
    //for(unsigned int i=0; i != hltbits_.size() ; ++i) {
    //  printBits(hltbits_[i]);
    //}
    //std::cout << "\n";
  }

// test luminosity sections
  void
  StreamerOutputModuleBase::setLumiSection() {
    struct timeval now;
    struct timezone dummyTZ;
    gettimeofday(&now, &dummyTZ);
    double timeInSec = static_cast<double>(now.tv_sec) + (static_cast<double>(now.tv_usec)/1000000.0) - timeInSecSinceUTC;
    // what about overflows?
    if(lumiSectionInterval_ > 0) lumi_ = static_cast<uint32>(timeInSec/lumiSectionInterval_) + 1;
  }

  std::auto_ptr<EventMsgBuilder>
  StreamerOutputModuleBase::serializeEvent(EventPrincipal const& e, ModuleCallingContext const* mcc) {
    //Lets Build the Event Message first

    //Following is strictly DUMMY Data for L! Trig and will be replaced with actual
    // once figured out, there is no logic involved here.
    l1bit_.push_back(true);
    l1bit_.push_back(true);
    l1bit_.push_back(false);
    //End of dummy data

    setHltMask(e, mcc);

    if (lumiSectionInterval_ == 0) {
      lumi_ = e.luminosityBlock();
    } else {
      setLumiSection();
    }

    serializer_.serializeEvent(e, selectorConfig(), useCompression_, compressionLevel_, serialize_databuffer, mcc);

    // resize bufs_ to reflect space used in serializer_ + header
    // I just added an overhead for header of 50000 for now
    unsigned int src_size = serialize_databuffer.currentSpaceUsed();
    unsigned int new_size = src_size + 50000;
    if(serialize_databuffer.bufs_.size() < new_size) serialize_databuffer.bufs_.resize(new_size);

    std::auto_ptr<EventMsgBuilder>
      msg(new EventMsgBuilder(&serialize_databuffer.bufs_[0], serialize_databuffer.bufs_.size(), e.id().run(),
                              e.id().event(), lumi_, outputModuleId_, 0,
                              l1bit_, (uint8*)&hltbits_[0], hltsize_,
                              (uint32)serialize_databuffer.adler32_chksum(), host_name_) );
    msg->setOrigDataSize(origSize_); // we need this set to zero

    // copy data into the destination message
    // an alternative is to have serializer only to the serialization
    // in serializeEvent, and then call a new member "getEventData" that
    // takes the compression arguments and a place to put the data.
    // This will require one less copy.  The only catch is that the
    // space provided in bufs_ should be at least the uncompressed
    // size + overhead for header because we will not know the actual
    // compressed size.

    unsigned char* src = serialize_databuffer.bufferPointer();
    std::copy(src,src + src_size, msg->eventAddr());
    msg->setEventLength(src_size);
    if(useCompression_) msg->setOrigDataSize(serialize_databuffer.currentEventSize());

    l1bit_.clear();  //Clear up for the next event to come.
    return msg;
  }

  void
  StreamerOutputModuleBase::fillDescription(ParameterSetDescription& desc) {
    desc.addUntracked<int>("max_event_size", 7000000)
        ->setComment("Starting size in bytes of the serialized event buffer.");
    desc.addUntracked<bool>("use_compression", true)
        ->setComment("If True, compression will be used to write streamer file.");
    desc.addUntracked<int>("compression_level", 1)
        ->setComment("ROOT compression level to use.");
    desc.addUntracked<int>("lumiSection_interval", 0)
        ->setComment("If 0, use lumi section number from event.\n"
                     "If not 0, the interval in seconds between fake lumi sections.");
    OutputModule::fillDescription(desc);
  }
} // end of namespace-edm

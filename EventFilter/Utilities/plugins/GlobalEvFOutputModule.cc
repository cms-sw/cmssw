#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/RunForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Utilities/interface/UnixSignalHandlers.h"

#include "FWCore/Concurrency/interface/SerialTaskQueue.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"

#include "IOPool/Streamer/interface/StreamerOutputFile.h"
#include "FWCore/Framework/interface/global/OutputModule.h"

#include "IOPool/Streamer/interface/StreamerOutputModuleCommon.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Streamer/interface/StreamedProducts.h"

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <sys/stat.h>
#include <filesystem>
#include <boost/algorithm/string.hpp>

typedef edm::detail::TriggerResultsBasedEventSelector::handle_t Trig;

namespace evf {

  class FastMonitoringService;

  class GlobalEvFOutputEventWriter {
  public:
    explicit GlobalEvFOutputEventWriter(std::string const& filePath, unsigned int ls)
        : filePath_(filePath), ls_(ls), accepted_(0), stream_writer_events_(new StreamerOutputFile(filePath)) {}

    ~GlobalEvFOutputEventWriter() {}

    bool close() {
      stream_writer_events_->close();
      return (discarded_ || edm::Service<evf::EvFDaqDirector>()->lumisectionDiscarded(ls_));
    }

    void doOutputEvent(EventMsgBuilder const& msg) {
      EventMsgView eview(msg.startAddress());
      stream_writer_events_->write(eview);
      incAccepted();
    }

    void doOutputEventAsync(std::unique_ptr<EventMsgBuilder> msg, edm::WaitingTaskHolder iHolder) {
      throttledCheck();
      discardedCheck();
      if (discarded_) {
        incAccepted();
        msg.reset();
        return;
      }
      auto group = iHolder.group();
      writeQueue_.push(*group, [holder = std::move(iHolder), msg = msg.release(), this]() {
        try {
          std::unique_ptr<EventMsgBuilder> own(msg);
          doOutputEvent(*msg);  //msg is written and discarded at this point
        } catch (...) {
          auto tmp = holder;
          tmp.doneWaiting(std::current_exception());
        }
      });
    }

    inline void throttledCheck() {
      unsigned int counter = 0;
      while (edm::Service<evf::EvFDaqDirector>()->inputThrottled() && !discarded_) {
        if (edm::shutdown_flag.load(std::memory_order_relaxed))
          break;
        if (!(counter % 100))
          edm::LogWarning("FedRawDataInputSource") << "Input throttled detected, writing is paused...";
        usleep(100000);
        counter++;
        if (edm::Service<evf::EvFDaqDirector>()->lumisectionDiscarded(ls_)) {
          edm::LogWarning("FedRawDataInputSource") << "Detected that the lumisection is discarded -: " << ls_;
          discarded_ = true;
        }
      }
    }

    inline void discardedCheck() {
      if (!discarded_ && edm::Service<evf::EvFDaqDirector>()->lumisectionDiscarded(ls_)) {
        edm::LogWarning("FedRawDataInputSource") << "Detected that the lumisection is discarded -: " << ls_;
        discarded_ = true;
      }
    }

    uint32 get_adler32() const { return stream_writer_events_->adler32(); }

    std::string const& getFilePath() const { return filePath_; }

    unsigned long getAccepted() const { return accepted_; }
    void incAccepted() { accepted_++; }

    edm::SerialTaskQueue& queue() { return writeQueue_; }

  private:
    std::string filePath_;
    const unsigned ls_;
    std::atomic<unsigned long> accepted_;
    edm::propagate_const<std::unique_ptr<StreamerOutputFile>> stream_writer_events_;
    edm::SerialTaskQueue writeQueue_;
    bool discarded_ = false;
  };

  class GlobalEvFOutputJSONDef {
  public:
    GlobalEvFOutputJSONDef(std::string const& streamLabel, bool writeJsd);
    void updateDestination(std::string const& streamLabel);

    jsoncollector::DataPointDefinition outJsonDef_;
    std::string outJsonDefName_;
    jsoncollector::StringJ transferDestination_;
    jsoncollector::StringJ mergeType_;
  };

  class GlobalEvFOutputJSONWriter {
  public:
    GlobalEvFOutputJSONWriter(std::string const& streamLabel,
                              jsoncollector::DataPointDefinition const&,
                              std::string const& outJsonDefName,
                              jsoncollector::StringJ const& transferDestination,
                              jsoncollector::StringJ const& mergeType);

    jsoncollector::IntJ processed_;
    jsoncollector::IntJ accepted_;
    jsoncollector::IntJ errorEvents_;
    jsoncollector::IntJ retCodeMask_;
    jsoncollector::StringJ filelist_;
    jsoncollector::IntJ filesize_;
    jsoncollector::StringJ inputFiles_;
    jsoncollector::IntJ fileAdler32_;
    jsoncollector::StringJ transferDestination_;
    jsoncollector::StringJ mergeType_;
    jsoncollector::IntJ hltErrorEvents_;
    std::shared_ptr<jsoncollector::FastMonitor> jsonMonitor_;
  };

  typedef edm::global::OutputModule<edm::RunCache<GlobalEvFOutputJSONDef>,
                                    edm::LuminosityBlockCache<evf::GlobalEvFOutputEventWriter>,
                                    edm::StreamCache<edm::StreamerOutputModuleCommon>,
                                    edm::ExternalWork>
      GlobalEvFOutputModuleType;

  class GlobalEvFOutputModule : public GlobalEvFOutputModuleType {
  public:
    explicit GlobalEvFOutputModule(edm::ParameterSet const& ps);
    ~GlobalEvFOutputModule() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::unique_ptr<edm::StreamerOutputModuleCommon> beginStream(edm::StreamID) const final;

    std::shared_ptr<GlobalEvFOutputJSONDef> globalBeginRun(edm::RunForOutput const& run) const final;

    void acquire(edm::StreamID, edm::EventForOutput const&, edm::WaitingTaskWithArenaHolder) const final;
    void write(edm::EventForOutput const& e) final;

    //pure in parent class but unused here
    void writeLuminosityBlock(edm::LuminosityBlockForOutput const&) final {}
    void writeRun(edm::RunForOutput const&) final {}
    void globalEndRun(edm::RunForOutput const&) const final {}

    std::shared_ptr<GlobalEvFOutputEventWriter> globalBeginLuminosityBlock(
        edm::LuminosityBlockForOutput const& iLB) const final;
    void globalEndLuminosityBlock(edm::LuminosityBlockForOutput const& iLB) const final;

    Trig getTriggerResults(edm::EDGetTokenT<edm::TriggerResults> const& token, edm::EventForOutput const& e) const;

    edm::StreamerOutputModuleCommon::Parameters commonParameters_;
    std::string streamLabel_;
    edm::EDGetTokenT<edm::TriggerResults> trToken_;
    edm::EDGetTokenT<edm::SendJobHeader::ParameterSetMap> psetToken_;

    evf::FastMonitoringService* fms_;

  };  //end-of-class-def

  GlobalEvFOutputJSONDef::GlobalEvFOutputJSONDef(std::string const& streamLabel, bool writeJsd) {
    std::string baseRunDir = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
    LogDebug("GlobalEvFOutputModule") << "writing .dat files to -: " << baseRunDir;

    outJsonDef_.setDefaultGroup("data");
    outJsonDef_.addLegendItem("Processed", "integer", jsoncollector::DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("Accepted", "integer", jsoncollector::DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("ErrorEvents", "integer", jsoncollector::DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("ReturnCodeMask", "integer", jsoncollector::DataPointDefinition::BINARYOR);
    outJsonDef_.addLegendItem("Filelist", "string", jsoncollector::DataPointDefinition::MERGE);
    outJsonDef_.addLegendItem("Filesize", "integer", jsoncollector::DataPointDefinition::SUM);
    outJsonDef_.addLegendItem("InputFiles", "string", jsoncollector::DataPointDefinition::CAT);
    outJsonDef_.addLegendItem("FileAdler32", "integer", jsoncollector::DataPointDefinition::ADLER32);
    outJsonDef_.addLegendItem("TransferDestination", "string", jsoncollector::DataPointDefinition::SAME);
    outJsonDef_.addLegendItem("MergeType", "string", jsoncollector::DataPointDefinition::SAME);
    outJsonDef_.addLegendItem("HLTErrorEvents", "integer", jsoncollector::DataPointDefinition::SUM);

    std::stringstream ss;
    ss << baseRunDir << "/"
       << "output_" << getpid() << ".jsd";
    outJsonDefName_ = ss.str();

    if (writeJsd) {
      std::stringstream tmpss;
      tmpss << baseRunDir << "/open/"
            << "output_" << getpid() << ".jsd";
      std::string outTmpJsonDefName = tmpss.str();
      edm::Service<evf::EvFDaqDirector>()->createRunOpendirMaybe();
      edm::Service<evf::EvFDaqDirector>()->lockInitLock();
      struct stat fstat;
      if (stat(outJsonDefName_.c_str(), &fstat) != 0) {  //file does not exist
        LogDebug("GlobalEvFOutputModule") << "writing output definition file -: " << outJsonDefName_;
        std::string content;
        jsoncollector::JSONSerializer::serialize(&outJsonDef_, content);
        jsoncollector::FileIO::writeStringToFile(outTmpJsonDefName, content);
        std::filesystem::rename(outTmpJsonDefName, outJsonDefName_);
      }
    }
    edm::Service<evf::EvFDaqDirector>()->unlockInitLock();
  }

  void GlobalEvFOutputJSONDef::updateDestination(std::string const& streamLabel) {
    transferDestination_ = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(streamLabel);
    mergeType_ = edm::Service<evf::EvFDaqDirector>()->getStreamMergeType(streamLabel, evf::MergeTypeDAT);
  }

  GlobalEvFOutputJSONWriter::GlobalEvFOutputJSONWriter(std::string const& streamLabel,
                                                       jsoncollector::DataPointDefinition const& outJsonDef,
                                                       std::string const& outJsonDefName,
                                                       jsoncollector::StringJ const& transferDestination,
                                                       jsoncollector::StringJ const& mergeType)
      : processed_(0),
        accepted_(0),
        errorEvents_(0),
        retCodeMask_(0),
        filelist_(),
        filesize_(0),
        inputFiles_(),
        fileAdler32_(1),
        transferDestination_(transferDestination),
        mergeType_(mergeType),
        hltErrorEvents_(0) {
    processed_.setName("Processed");
    accepted_.setName("Accepted");
    errorEvents_.setName("ErrorEvents");
    retCodeMask_.setName("ReturnCodeMask");
    filelist_.setName("Filelist");
    filesize_.setName("Filesize");
    inputFiles_.setName("InputFiles");
    fileAdler32_.setName("FileAdler32");
    transferDestination_.setName("TransferDestination");
    mergeType_.setName("MergeType");
    hltErrorEvents_.setName("HLTErrorEvents");

    jsonMonitor_.reset(new jsoncollector::FastMonitor(&outJsonDef, true));
    jsonMonitor_->setDefPath(outJsonDefName);
    jsonMonitor_->registerGlobalMonitorable(&processed_, false);
    jsonMonitor_->registerGlobalMonitorable(&accepted_, false);
    jsonMonitor_->registerGlobalMonitorable(&errorEvents_, false);
    jsonMonitor_->registerGlobalMonitorable(&retCodeMask_, false);
    jsonMonitor_->registerGlobalMonitorable(&filelist_, false);
    jsonMonitor_->registerGlobalMonitorable(&filesize_, false);
    jsonMonitor_->registerGlobalMonitorable(&inputFiles_, false);
    jsonMonitor_->registerGlobalMonitorable(&fileAdler32_, false);
    jsonMonitor_->registerGlobalMonitorable(&transferDestination_, false);
    jsonMonitor_->registerGlobalMonitorable(&mergeType_, false);
    jsonMonitor_->registerGlobalMonitorable(&hltErrorEvents_, false);
    jsonMonitor_->commit(nullptr);
  }

  GlobalEvFOutputModule::GlobalEvFOutputModule(edm::ParameterSet const& ps)
      : edm::global::OutputModuleBase(ps),
        GlobalEvFOutputModuleType(ps),
        commonParameters_(edm::StreamerOutputModuleCommon::parameters(ps)),
        streamLabel_(ps.getParameter<std::string>("@module_label")),
        trToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"))),
        psetToken_(consumes<edm::SendJobHeader::ParameterSetMap, edm::InRun>(
            ps.getUntrackedParameter<edm::InputTag>("psetMap"))) {
    //replace hltOutoputA with stream if the HLT menu uses this convention
    std::string testPrefix = "hltOutput";
    if (streamLabel_.find(testPrefix) == 0)
      streamLabel_ = std::string("stream") + streamLabel_.substr(testPrefix.size());

    if (streamLabel_.find('_') != std::string::npos) {
      throw cms::Exception("GlobalEvFOutputModule")
          << "Underscore character is reserved can not be used for stream names in "
             "FFF, but was detected in stream name -: "
          << streamLabel_;
    }

    std::string streamLabelLow = streamLabel_;
    boost::algorithm::to_lower(streamLabelLow);
    auto streampos = streamLabelLow.rfind("stream");
    if (streampos != 0 && streampos != std::string::npos)
      throw cms::Exception("GlobalEvFOutputModule")
          << "stream (case-insensitive) sequence was found in stream suffix. This is reserved and can not be used for "
             "names in FFF based HLT, but was detected in stream name";

    //output initemp file. This lets hltd know number of streams early on
    if (!edm::Service<evf::EvFDaqDirector>().isAvailable())
      throw cms::Exception("GlobalEvFOutputModule") << "EvFDaqDirector is not available";

    const std::string iniFileName = edm::Service<evf::EvFDaqDirector>()->getInitTempFilePath(streamLabel_);
    std::ofstream file(iniFileName);
    if (!file)
      throw cms::Exception("GlobalEvFOutputModule") << "can not create " << iniFileName << "error: " << strerror(errno);
    file.close();

    edm::LogInfo("GlobalEvFOutputModule") << "Constructor created initemp file -: " << iniFileName;

    //create JSD
    GlobalEvFOutputJSONDef(streamLabel_, true);

    fms_ = (evf::FastMonitoringService*)(edm::Service<evf::MicroStateService>().operator->());
  }

  GlobalEvFOutputModule::~GlobalEvFOutputModule() {}

  void GlobalEvFOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::StreamerOutputModuleCommon::fillDescription(desc);
    GlobalEvFOutputModuleType::fillDescription(desc);
    desc.addUntracked<edm::InputTag>("psetMap", {"hltPSetMap"})
        ->setComment("Optionally allow the map of ParameterSets to be calculated externally.");
    descriptions.add("globalEvfOutputModule", desc);
  }

  std::unique_ptr<edm::StreamerOutputModuleCommon> GlobalEvFOutputModule::beginStream(edm::StreamID) const {
    return std::make_unique<edm::StreamerOutputModuleCommon>(
        commonParameters_, &keptProducts()[edm::InEvent], description().moduleLabel());
  }

  std::shared_ptr<GlobalEvFOutputJSONDef> GlobalEvFOutputModule::globalBeginRun(edm::RunForOutput const& run) const {
    //create run Cache holding JSON file writer and variables
    auto jsonDef = std::make_unique<GlobalEvFOutputJSONDef>(streamLabel_, false);
    jsonDef->updateDestination(streamLabel_);
    edm::StreamerOutputModuleCommon streamerCommon(
        commonParameters_, &keptProducts()[edm::InEvent], description().moduleLabel());

    //output INI file (non-const). This doesn't require globalBeginRun to be finished
    const std::string openIniFileName = edm::Service<evf::EvFDaqDirector>()->getOpenInitFilePath(streamLabel_);
    edm::LogInfo("GlobalEvFOutputModule") << "beginRun init stream -: " << openIniFileName;

    StreamerOutputFile stream_writer_preamble(openIniFileName);
    uint32 preamble_adler32 = 1;
    edm::BranchIDLists const* bidlPtr = branchIDLists();

    auto psetMapHandle = run.getHandle(psetToken_);

    std::unique_ptr<InitMsgBuilder> init_message =
        streamerCommon.serializeRegistry(*streamerCommon.getSerializerBuffer(),
                                         *bidlPtr,
                                         *thinnedAssociationsHelper(),
                                         OutputModule::processName(),
                                         description().moduleLabel(),
                                         moduleDescription().mainParameterSetID(),
                                         psetMapHandle.isValid() ? psetMapHandle.product() : nullptr);

    //Let us turn it into a View
    InitMsgView view(init_message->startAddress());

    //output header
    stream_writer_preamble.write(view);
    preamble_adler32 = stream_writer_preamble.adler32();
    stream_writer_preamble.close();

    struct stat istat;
    stat(openIniFileName.c_str(), &istat);
    //read back file to check integrity of what was written
    off_t readInput = 0;
    uint32_t adlera = 1, adlerb = 0;
    std::ifstream src(openIniFileName, std::ifstream::binary);
    if (!src)
      throw cms::Exception("GlobalEvFOutputModule")
          << "can not read back " << openIniFileName << " error: " << strerror(errno);

    //allocate buffer to write INI file
    std::unique_ptr<char[]> outBuf = std::make_unique<char[]>(1024 * 1024);
    while (readInput < istat.st_size) {
      size_t toRead = readInput + 1024 * 1024 < istat.st_size ? 1024 * 1024 : istat.st_size - readInput;
      src.read(outBuf.get(), toRead);
      //cms::Adler32(const_cast<const char*>(reinterpret_cast<char*>(outBuf.get())), toRead, adlera, adlerb);
      cms::Adler32(const_cast<const char*>(outBuf.get()), toRead, adlera, adlerb);
      readInput += toRead;
    }
    src.close();

    //clear serialization buffers
    streamerCommon.getSerializerBuffer()->clearHeaderBuffer();

    //free output buffer needed only for the file write
    outBuf.reset();

    uint32_t adler32c = (adlerb << 16) | adlera;
    if (adler32c != preamble_adler32) {
      throw cms::Exception("GlobalEvFOutputModule") << "Checksum mismatch of ini file -: " << openIniFileName
                                                    << " expected:" << preamble_adler32 << " obtained:" << adler32c;
    } else {
      LogDebug("GlobalEvFOutputModule") << "Ini file checksum -: " << streamLabel_ << " " << adler32c;
      std::filesystem::rename(openIniFileName, edm::Service<evf::EvFDaqDirector>()->getInitFilePath(streamLabel_));
    }

    return jsonDef;
  }

  Trig GlobalEvFOutputModule::getTriggerResults(edm::EDGetTokenT<edm::TriggerResults> const& token,
                                                edm::EventForOutput const& e) const {
    Trig result;
    e.getByToken<edm::TriggerResults>(token, result);
    return result;
  }

  std::shared_ptr<GlobalEvFOutputEventWriter> GlobalEvFOutputModule::globalBeginLuminosityBlock(
      edm::LuminosityBlockForOutput const& iLB) const {
    auto openDatFilePath = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(iLB.luminosityBlock(), streamLabel_);

    return std::make_shared<GlobalEvFOutputEventWriter>(openDatFilePath, iLB.luminosityBlock());
  }

  void GlobalEvFOutputModule::acquire(edm::StreamID id,
                                      edm::EventForOutput const& e,
                                      edm::WaitingTaskWithArenaHolder iHolder) const {
    edm::Handle<edm::TriggerResults> const& triggerResults = getTriggerResults(trToken_, e);

    auto streamerCommon = streamCache(id);
    std::unique_ptr<EventMsgBuilder> msg =
        streamerCommon->serializeEvent(*streamerCommon->getSerializerBuffer(), e, triggerResults, selectorConfig());

    auto lumiWriter = luminosityBlockCache(e.getLuminosityBlock().index());
    const_cast<evf::GlobalEvFOutputEventWriter*>(lumiWriter)
        ->doOutputEventAsync(std::move(msg), iHolder.makeWaitingTaskHolderAndRelease());
  }
  void GlobalEvFOutputModule::write(edm::EventForOutput const&) {}

  void GlobalEvFOutputModule::globalEndLuminosityBlock(edm::LuminosityBlockForOutput const& iLB) const {
    auto lumiWriter = luminosityBlockCache(iLB.index());
    //close dat file
    const bool discarded = const_cast<evf::GlobalEvFOutputEventWriter*>(lumiWriter)->close();

    //auto jsonWriter = const_cast<GlobalEvFOutputJSONWriter*>(runCache(iLB.getRun().index()));
    auto jsonDef = runCache(iLB.getRun().index());
    GlobalEvFOutputJSONWriter jsonWriter(streamLabel_,
                                         jsonDef->outJsonDef_,
                                         jsonDef->outJsonDefName_,
                                         jsonDef->transferDestination_,
                                         jsonDef->mergeType_);

    jsonWriter.fileAdler32_.value() = lumiWriter->get_adler32();
    jsonWriter.accepted_.value() = lumiWriter->getAccepted();

    bool abortFlag = false;

    if (!discarded) {
      jsonWriter.processed_.value() = fms_->getEventsProcessedForLumi(iLB.luminosityBlock(), &abortFlag);
    } else {
      jsonWriter.errorEvents_.value() = fms_->getEventsProcessedForLumi(iLB.luminosityBlock(), &abortFlag);
      jsonWriter.processed_.value() = 0;
      jsonWriter.accepted_.value() = 0;
      edm::LogInfo("GlobalEvFOutputModule")
          << "Output suppressed, setting error events for LS -: " << iLB.luminosityBlock();
    }

    if (abortFlag) {
      edm::LogInfo("GlobalEvFOutputModule") << "Abort flag has been set. Output is suppressed";
      return;
    }

    if (jsonWriter.processed_.value() != 0) {
      struct stat istat;
      std::filesystem::path openDatFilePath = lumiWriter->getFilePath();
      stat(openDatFilePath.string().c_str(), &istat);
      jsonWriter.filesize_ = istat.st_size;
      std::filesystem::rename(openDatFilePath.string().c_str(),
                              edm::Service<evf::EvFDaqDirector>()->getDatFilePath(iLB.luminosityBlock(), streamLabel_));
      jsonWriter.filelist_ = openDatFilePath.filename().string();
    } else {
      //remove empty file when no event processing has occurred
      remove(lumiWriter->getFilePath().c_str());
      jsonWriter.filesize_ = 0;
      jsonWriter.filelist_ = "";
      jsonWriter.fileAdler32_.value() = -1;  //no files in signed long
    }

    //produce JSON file
    jsonWriter.jsonMonitor_->snap(iLB.luminosityBlock());
    const std::string outputJsonNameStream =
        edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(iLB.luminosityBlock(), streamLabel_);
    jsonWriter.jsonMonitor_->outputFullJSON(outputJsonNameStream, iLB.luminosityBlock());
  }

}  // namespace evf

using namespace evf;
DEFINE_FWK_MODULE(GlobalEvFOutputModule);

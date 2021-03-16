#include "EventFilter/Utilities/interface/EvFOutputModule.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "FWCore/Framework/interface/EventForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"

#include <sys/stat.h>
#include <filesystem>
#include <boost/algorithm/string.hpp>

namespace evf {

  EvFOutputJSONWriter::EvFOutputJSONWriter(edm::ParameterSet const& ps,
                                           edm::SelectedProducts const* selections,
                                           std::string const& streamLabel)
      : streamerCommon_(ps, selections),
        processed_(0),
        accepted_(0),
        errorEvents_(0),
        retCodeMask_(0),
        filelist_(),
        filesize_(0),
        inputFiles_(),
        fileAdler32_(1),
        hltErrorEvents_(0) {
    transferDestination_ = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(streamLabel);
    mergeType_ = edm::Service<evf::EvFDaqDirector>()->getStreamMergeType(streamLabel, evf::MergeTypeDAT);

    std::string baseRunDir = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
    LogDebug("EvFOutputModule") << "writing .dat files to -: " << baseRunDir;

    edm::Service<evf::EvFDaqDirector>()->createRunOpendirMaybe();

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

    std::stringstream tmpss, ss;
    tmpss << baseRunDir << "/open/"
          << "output_" << getpid() << ".jsd";
    ss << baseRunDir << "/"
       << "output_" << getpid() << ".jsd";
    std::string outTmpJsonDefName = tmpss.str();
    std::string outJsonDefName = ss.str();

    edm::Service<evf::EvFDaqDirector>()->lockInitLock();
    struct stat fstat;
    if (stat(outJsonDefName.c_str(), &fstat) != 0) {  //file does not exist
      LogDebug("EvFOutputModule") << "writing output definition file -: " << outJsonDefName;
      std::string content;
      jsoncollector::JSONSerializer::serialize(&outJsonDef_, content);
      jsoncollector::FileIO::writeStringToFile(outTmpJsonDefName, content);
      std::filesystem::rename(outTmpJsonDefName, outJsonDefName);
    }
    edm::Service<evf::EvFDaqDirector>()->unlockInitLock();

    jsonMonitor_.reset(new jsoncollector::FastMonitor(&outJsonDef_, true));
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

  EvFOutputModule::EvFOutputModule(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase(ps),
        EvFOutputModuleType(ps),
        ps_(ps),
        streamLabel_(ps.getParameter<std::string>("@module_label")),
        trToken_(consumes<edm::TriggerResults>(edm::InputTag("TriggerResults"))),
        psetToken_(consumes<edm::SendJobHeader::ParameterSetMap, edm::InRun>(
            ps.getUntrackedParameter<edm::InputTag>("psetMap"))) {
    //replace hltOutoputA with stream if the HLT menu uses this convention
    std::string testPrefix = "hltOutput";
    if (streamLabel_.find(testPrefix) == 0)
      streamLabel_ = std::string("stream") + streamLabel_.substr(testPrefix.size());

    if (streamLabel_.find('_') != std::string::npos) {
      throw cms::Exception("EvFOutputModule") << "Underscore character is reserved can not be used for stream names in "
                                                 "FFF, but was detected in stream name -: "
                                              << streamLabel_;
    }

    std::string streamLabelLow = streamLabel_;
    boost::algorithm::to_lower(streamLabelLow);
    auto streampos = streamLabelLow.rfind("stream");
    if (streampos != 0 && streampos != std::string::npos)
      throw cms::Exception("EvFOutputModule")
          << "stream (case-insensitive) sequence was found in stream suffix. This is reserved and can not be used for "
             "names in FFF based HLT, but was detected in stream name";

    fms_ = (evf::FastMonitoringService*)(edm::Service<evf::MicroStateService>().operator->());
  }

  EvFOutputModule::~EvFOutputModule() {}

  void EvFOutputModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::StreamerOutputModuleCommon::fillDescription(desc);
    EvFOutputModuleType::fillDescription(desc);
    desc.addUntracked<edm::InputTag>("psetMap", {"hltPSetMap"})
        ->setComment("Optionally allow the map of ParameterSets to be calculated externally.");
    descriptions.add("evfOutputModule", desc);
  }

  void EvFOutputModule::beginRun(edm::RunForOutput const& run) {
    //create run Cache holding JSON file writer and variables
    jsonWriter_ = std::make_unique<EvFOutputJSONWriter>(ps_, &keptProducts()[edm::InEvent], streamLabel_);

    //output INI file (non-const). This doesn't require globalBeginRun to be finished
    const std::string openIniFileName = edm::Service<evf::EvFDaqDirector>()->getOpenInitFilePath(streamLabel_);
    edm::LogInfo("EvFOutputModule") << "beginRun init stream -: " << openIniFileName;

    StreamerOutputFile stream_writer_preamble(openIniFileName);
    uint32 preamble_adler32 = 1;
    edm::BranchIDLists const* bidlPtr = branchIDLists();

    auto psetMapHandle = run.getHandle(psetToken_);

    std::unique_ptr<InitMsgBuilder> init_message =
        jsonWriter_->streamerCommon_.serializeRegistry(*jsonWriter_->streamerCommon_.getSerializerBuffer(),
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
    FILE* src = fopen(openIniFileName.c_str(), "r");

    //allocate buffer to write INI file
    unsigned char* outBuf = new unsigned char[1024 * 1024];
    while (readInput < istat.st_size) {
      size_t toRead = readInput + 1024 * 1024 < istat.st_size ? 1024 * 1024 : istat.st_size - readInput;
      fread(outBuf, toRead, 1, src);
      cms::Adler32((const char*)outBuf, toRead, adlera, adlerb);
      readInput += toRead;
    }
    fclose(src);

    //clear serialization buffers
    jsonWriter_->streamerCommon_.getSerializerBuffer()->clearHeaderBuffer();

    //free output buffer needed only for the file write
    delete[] outBuf;
    outBuf = nullptr;

    uint32_t adler32c = (adlerb << 16) | adlera;
    if (adler32c != preamble_adler32) {
      throw cms::Exception("EvFOutputModule") << "Checksum mismatch of ini file -: " << openIniFileName
                                              << " expected:" << preamble_adler32 << " obtained:" << adler32c;
    } else {
      LogDebug("EvFOutputModule") << "Ini file checksum -: " << streamLabel_ << " " << adler32c;
      std::filesystem::rename(openIniFileName, edm::Service<evf::EvFDaqDirector>()->getInitFilePath(streamLabel_));
    }
  }

  Trig EvFOutputModule::getTriggerResults(edm::EDGetTokenT<edm::TriggerResults> const& token,
                                          edm::EventForOutput const& e) const {
    Trig result;
    e.getByToken<edm::TriggerResults>(token, result);
    return result;
  }

  std::shared_ptr<EvFOutputEventWriter> EvFOutputModule::globalBeginLuminosityBlock(
      edm::LuminosityBlockForOutput const& iLB) const {
    auto openDatFilePath = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(iLB.luminosityBlock(), streamLabel_);

    return std::make_shared<EvFOutputEventWriter>(openDatFilePath);
  }

  void EvFOutputModule::write(edm::EventForOutput const& e) {
    edm::Handle<edm::TriggerResults> const& triggerResults = getTriggerResults(trToken_, e);

    //auto lumiWriter = const_cast<EvFOutputEventWriter*>(luminosityBlockCache(e.getLuminosityBlock().index() ));
    auto lumiWriter = luminosityBlockCache(e.getLuminosityBlock().index());
    std::unique_ptr<EventMsgBuilder> msg = jsonWriter_->streamerCommon_.serializeEvent(
        *jsonWriter_->streamerCommon_.getSerializerBuffer(), e, triggerResults, selectorConfig());
    lumiWriter->incAccepted();
    lumiWriter->doOutputEvent(*msg);  //msg is written and discarded at this point
  }

  void EvFOutputModule::globalEndLuminosityBlock(edm::LuminosityBlockForOutput const& iLB) {
    auto lumiWriter = luminosityBlockCache(iLB.index());
    //close dat file
    lumiWriter->close();

    jsonWriter_->fileAdler32_.value() = lumiWriter->get_adler32();
    jsonWriter_->accepted_.value() = lumiWriter->getAccepted();

    bool abortFlag = false;
    jsonWriter_->processed_.value() = fms_->getEventsProcessedForLumi(iLB.luminosityBlock(), &abortFlag);
    if (abortFlag) {
      edm::LogInfo("EvFOutputModule") << "Abort flag has been set. Output is suppressed";
      return;
    }

    if (jsonWriter_->processed_.value() != 0) {
      struct stat istat;
      std::filesystem::path openDatFilePath = lumiWriter->getFilePath();
      stat(openDatFilePath.string().c_str(), &istat);
      jsonWriter_->filesize_ = istat.st_size;
      std::filesystem::rename(openDatFilePath.string().c_str(),
                              edm::Service<evf::EvFDaqDirector>()->getDatFilePath(iLB.luminosityBlock(), streamLabel_));
      jsonWriter_->filelist_ = openDatFilePath.filename().string();
    } else {
      //remove empty file when no event processing has occurred
      remove(lumiWriter->getFilePath().c_str());
      jsonWriter_->filesize_ = 0;
      jsonWriter_->filelist_ = "";
      jsonWriter_->fileAdler32_.value() = -1;  //no files in signed long
    }

    //produce JSON file
    jsonWriter_->jsonMonitor_->snap(iLB.luminosityBlock());
    const std::string outputJsonNameStream =
        edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(iLB.luminosityBlock(), streamLabel_);
    jsonWriter_->jsonMonitor_->outputFullJSON(outputJsonNameStream, iLB.luminosityBlock());
  }

}  // namespace evf

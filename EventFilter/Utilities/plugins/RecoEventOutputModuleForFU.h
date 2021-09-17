#ifndef IOPool_Streamer_RecoEventOutputModuleForFU_h
#define IOPool_Streamer_RecoEventOutputModuleForFU_h

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "IOPool/Streamer/interface/StreamerOutputModuleBase.h"
#include "FWCore/Framework/interface/LuminosityBlockForOutput.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <filesystem>
#include <iomanip>
#include <sstream>

#include <zlib.h>
#include <boost/algorithm/string.hpp>

#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

namespace evf {
  template <typename Consumer>
  class RecoEventOutputModuleForFU : public edm::StreamerOutputModuleBase {
    /** Consumers are supposed to provide
	void doOutputHeader(InitMsgBuilder const& init_message)
	void doOutputEvent(EventMsgBuilder const& msg)
	void start()
	void stop()
	static void fillDescription(ParameterSetDescription&)
    **/

  public:
    explicit RecoEventOutputModuleForFU(edm::ParameterSet const& ps);
    ~RecoEventOutputModuleForFU() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void initRun();
    void start() override;
    void stop() override;
    void doOutputHeader(InitMsgBuilder const& init_message) override;
    void doOutputEvent(EventMsgBuilder const& msg) override;
    //virtual void beginRun(edm::RunForOutput const&);
    void beginJob() override;
    void beginLuminosityBlock(edm::LuminosityBlockForOutput const&) override;
    void endLuminosityBlock(edm::LuminosityBlockForOutput const&) override;

  private:
    std::unique_ptr<Consumer> c_;
    std::string streamLabel_;
    std::filesystem::path openDatFilePath_;
    std::filesystem::path openDatChecksumFilePath_;
    jsoncollector::IntJ processed_;
    mutable jsoncollector::IntJ accepted_;
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
    evf::FastMonitoringService* fms_;
    jsoncollector::DataPointDefinition outJsonDef_;
    unsigned char* outBuf_ = nullptr;
    bool readAdler32Check_ = false;
  };  //end-of-class-def

  template <typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::RecoEventOutputModuleForFU(edm::ParameterSet const& ps)
      : edm::one::OutputModuleBase::OutputModuleBase(ps),
        edm::StreamerOutputModuleBase(ps),
        c_(new Consumer(ps)),
        streamLabel_(ps.getParameter<std::string>("@module_label")),
        processed_(0),
        accepted_(0),
        errorEvents_(0),
        retCodeMask_(0),
        filelist_(),
        filesize_(0),
        inputFiles_(),
        fileAdler32_(1),
        transferDestination_(),
        mergeType_(),
        hltErrorEvents_(0),
        outBuf_(new unsigned char[1024 * 1024]) {
    //replace hltOutoputA with stream if the HLT menu uses this convention
    std::string testPrefix = "hltOutput";
    if (streamLabel_.find(testPrefix) == 0)
      streamLabel_ = std::string("stream") + streamLabel_.substr(testPrefix.size());

    if (streamLabel_.find('_') != std::string::npos) {
      throw cms::Exception("RecoEventOutputModuleForFU") << "Underscore character is reserved can not be used for "
                                                            "stream names in FFF, but was detected in stream name -: "
                                                         << streamLabel_;
    }

    std::string streamLabelLow = streamLabel_;
    boost::algorithm::to_lower(streamLabelLow);
    auto streampos = streamLabelLow.rfind("stream");
    if (streampos != 0 && streampos != std::string::npos)
      throw cms::Exception("RecoEventOutputModuleForFU")
          << "stream (case-insensitive) sequence was found in stream suffix. This is reserved and can not be used for "
             "names in FFF based HLT, but was detected in stream name";

    fms_ = (evf::FastMonitoringService*)(edm::Service<evf::MicroStateService>().operator->());
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::initRun() {
    std::string baseRunDir = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
    readAdler32Check_ = edm::Service<evf::EvFDaqDirector>()->outputAdler32Recheck();
    LogDebug("RecoEventOutputModuleForFU") << "writing .dat files to -: " << baseRunDir;
    // create open dir if not already there
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
      LogDebug("RecoEventOutputModuleForFU") << "writing output definition file -: " << outJsonDefName;
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

  template <typename Consumer>
  RecoEventOutputModuleForFU<Consumer>::~RecoEventOutputModuleForFU() {}

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::start() {
    initRun();
    const std::string openInitFileName = edm::Service<evf::EvFDaqDirector>()->getOpenInitFilePath(streamLabel_);
    edm::LogInfo("RecoEventOutputModuleForFU")
        << "start() method, initializing streams. init stream -: " << openInitFileName;
    c_->setInitMessageFile(openInitFileName);
    c_->start();
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::stop() {
    c_->stop();
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::doOutputHeader(InitMsgBuilder const& init_message) {
    c_->doOutputHeader(init_message);

    const std::string openIniFileName = edm::Service<evf::EvFDaqDirector>()->getOpenInitFilePath(streamLabel_);
    struct stat istat;
    stat(openIniFileName.c_str(), &istat);
    //read back file to check integrity of what was written
    off_t readInput = 0;
    uint32_t adlera = 1, adlerb = 0;
    FILE* src = fopen(openIniFileName.c_str(), "r");
    while (readInput < istat.st_size) {
      size_t toRead = readInput + 1024 * 1024 < istat.st_size ? 1024 * 1024 : istat.st_size - readInput;
      fread(outBuf_, toRead, 1, src);
      cms::Adler32((const char*)outBuf_, toRead, adlera, adlerb);
      readInput += toRead;
    }
    fclose(src);
    //free output buffer needed only for the INI file
    delete[] outBuf_;
    outBuf_ = nullptr;

    uint32_t adler32c = (adlerb << 16) | adlera;
    if (adler32c != c_->get_adler32_ini()) {
      throw cms::Exception("RecoEventOutputModuleForFU")
          << "Checksum mismatch of ini file -: " << openIniFileName << " expected:" << c_->get_adler32_ini()
          << " obtained:" << adler32c;
    } else {
      LogDebug("RecoEventOutputModuleForFU") << "Ini file checksum -: " << streamLabel_ << " " << adler32c;
      std::filesystem::rename(openIniFileName, edm::Service<evf::EvFDaqDirector>()->getInitFilePath(streamLabel_));
    }
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::doOutputEvent(EventMsgBuilder const& msg) {
    accepted_.value()++;
    c_->doOutputEvent(msg);  // You can't use msg in RecoEventOutputModuleForFU after this point
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::StreamerOutputModuleBase::fillDescription(desc);
    Consumer::fillDescription(desc);
    // Use addDefault here instead of add for 4 reasons:
    // 1. Because EvFOutputModule_cfi.py is explicitly defined it does not need to be autogenerated
    // The explicitly defined version overrides the autogenerated version of the cfi file.
    // 2. That cfi file is not used anywhere in the release anyway
    // 3. There are two plugin names used for the same template instantiation of this
    // type, "ShmStreamConsumer" and "EvFOutputModule" and this causes name conflict
    // problems for the cfi generation code which are avoided with addDefault.
    // 4. At the present time, there is only one type of Consumer used to instantiate
    // instances of this template, but if there were more than one type then this function
    // would need to be specialized for each type unless the descriptions were the same
    // and addDefault was used.
    descriptions.addDefault(desc);
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::beginJob() {
    //get stream transfer destination
    transferDestination_ = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(streamLabel_);
    mergeType_ = edm::Service<evf::EvFDaqDirector>()->getStreamMergeType(streamLabel_, evf::MergeTypeDAT);
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::beginLuminosityBlock(edm::LuminosityBlockForOutput const& ls) {
    //edm::LogInfo("RecoEventOutputModuleForFU") << "begin lumi";
    openDatFilePath_ = edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(ls.luminosityBlock(), streamLabel_);
    openDatChecksumFilePath_ =
        edm::Service<evf::EvFDaqDirector>()->getOpenDatFilePath(ls.luminosityBlock(), streamLabel_);
    c_->setOutputFile(openDatFilePath_.string());
    filelist_ = openDatFilePath_.filename().string();
  }

  template <typename Consumer>
  void RecoEventOutputModuleForFU<Consumer>::endLuminosityBlock(edm::LuminosityBlockForOutput const& ls) {
    //edm::LogInfo("RecoEventOutputModuleForFU") << "end lumi";
    long filesize = 0;
    fileAdler32_.value() = c_->get_adler32();
    c_->closeOutputFile();
    bool abortFlag = false;
    processed_.value() = fms_->getEventsProcessedForLumi(ls.luminosityBlock(), &abortFlag);

    if (abortFlag) {
      edm::LogInfo("RecoEventOutputModuleForFU") << "output suppressed";
      return;
    }

    if (processed_.value() != 0) {
      //lock
      struct stat istat;
      stat(openDatFilePath_.string().c_str(), &istat);
      filesize = istat.st_size;
      std::filesystem::rename(openDatFilePath_.string().c_str(),
                              edm::Service<evf::EvFDaqDirector>()->getDatFilePath(ls.luminosityBlock(), streamLabel_));
    } else {
      filelist_ = "";
      fileAdler32_.value() = -1;
    }

    //remove file
    remove(openDatFilePath_.string().c_str());
    filesize_ = filesize;

    jsonMonitor_->snap(ls.luminosityBlock());
    const std::string outputJsonNameStream =
        edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(ls.luminosityBlock(), streamLabel_);
    jsonMonitor_->outputFullJSON(outputJsonNameStream, ls.luminosityBlock());

    // reset monitoring params
    accepted_.value() = 0;
    filelist_ = "";
  }

}  // namespace evf

#endif

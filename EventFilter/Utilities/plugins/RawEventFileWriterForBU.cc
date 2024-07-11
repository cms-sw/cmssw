#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/FRDFileHeader.h"

using namespace jsoncollector;
using namespace edm::streamer;

//TODO:get run directory information from DaqDirector

RawEventFileWriterForBU::RawEventFileWriterForBU(edm::ParameterSet const& ps)
    : microSleep_(ps.getParameter<int>("microSleep")),
      frdFileVersion_(ps.getParameter<unsigned int>("frdFileVersion")) {

  if (edm::Service<evf::FastMonitoringService>().isAvailable())
    fms_ = static_cast<evf::FastMonitoringService*>(edm::Service<evf::FastMonitoringService>().operator->());

  //per-file JSD and FastMonitor
  rawJsonDef_.setDefaultGroup("legend");
  rawJsonDef_.addLegendItem("NEvents", "integer", DataPointDefinition::SUM);

  perFileEventCount_.setName("NEvents");
  perFileSize_.setName("NBytes");

  fileMon_ = new FastMonitor(&rawJsonDef_, false);
  fileMon_->registerGlobalMonitorable(&perFileEventCount_, false, nullptr);
  fileMon_->registerGlobalMonitorable(&perFileSize_, false, nullptr);
  fileMon_->commit(nullptr);

  //per-lumi JSD and FastMonitor
  eolJsonDef_.setDefaultGroup("legend");
  eolJsonDef_.addLegendItem("NEvents", "integer", DataPointDefinition::SUM);
  eolJsonDef_.addLegendItem("NFiles", "integer", DataPointDefinition::SUM);
  eolJsonDef_.addLegendItem("TotalEvents", "integer", DataPointDefinition::SUM);
  eolJsonDef_.addLegendItem("NLostEvents", "integer", DataPointDefinition::SUM);
  eolJsonDef_.addLegendItem("NBytes", "integer", DataPointDefinition::SUM);

  perLumiEventCount_.setName("NEvents");
  perLumiFileCount_.setName("NFiles");
  perLumiTotalEventCount_.setName("TotalEvents");
  perLumiLostEventCount_.setName("NLostEvents");
  perLumiSize_.setName("NBytes");

  lumiMon_ = new FastMonitor(&eolJsonDef_, false);
  lumiMon_->registerGlobalMonitorable(&perLumiEventCount_, false, nullptr);
  lumiMon_->registerGlobalMonitorable(&perLumiFileCount_, false, nullptr);
  lumiMon_->registerGlobalMonitorable(&perLumiTotalEventCount_, false, nullptr);
  lumiMon_->registerGlobalMonitorable(&perLumiLostEventCount_, false, nullptr);
  lumiMon_->registerGlobalMonitorable(&perLumiSize_, false, nullptr);
  lumiMon_->commit(nullptr);

  //per-run JSD and FastMonitor
  eorJsonDef_.setDefaultGroup("legend");
  eorJsonDef_.addLegendItem("NEvents", "integer", DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("NFiles", "integer", DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("NLumis", "integer", DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("LastLumi", "integer", DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("TotalEvents", "integer", DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("NLostEvents", "integer", DataPointDefinition::SUM);

  perRunEventCount_.setName("NEvents");
  perRunFileCount_.setName("NFiles");
  perRunLumiCount_.setName("NLumis");
  perRunLastLumi_.setName("LastLumi");
  perRunTotalEventCount_.setName("TotalEvents");
  perRunLostEventCount_.setName("NLostEvents");

  runMon_ = new FastMonitor(&eorJsonDef_, false);
  runMon_->registerGlobalMonitorable(&perRunEventCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunFileCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunLumiCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunLastLumi_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunTotalEventCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunLostEventCount_, false, nullptr);

  runMon_->commit(nullptr);
}

RawEventFileWriterForBU::RawEventFileWriterForBU(std::string const& fileName) {}

RawEventFileWriterForBU::~RawEventFileWriterForBU() {
  delete fileMon_;
  delete lumiMon_;
  delete runMon_;
}

void RawEventFileWriterForBU::doOutputEvent(FRDEventMsgView const& msg) {
  ssize_t retval = write(outfd_, (void*)msg.startAddress(), msg.size());

  if ((unsigned)retval != msg.size()) {
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEvent")
        << "Error writing FED Raw Data event data to " << fileName_ << ".  Possibly the output disk "
        << "is full?" << std::endl;
  }

  // throttle event output
  usleep(microSleep_);
  perFileEventCount_.value()++;
  perFileSize_.value() += msg.size();

  //  cms::Adler32((const char*) msg.startAddress(), msg.size(), adlera_, adlerb_);
}

void RawEventFileWriterForBU::initialize(std::string const& destinationDir, std::string const& name, int run, unsigned int ls) {
  destinationDir_ = destinationDir;
  run_ = run;

  std::stringstream ss;
  ss << "run" << std::setfill('0') << std::setw(6) << run_;
  runPrefix_ = ss.str();

  if (outfd_ != -1) {
    if (!fms_ || !fms_->exceptionDetected() || !fms_->getAbortFlagForLumi(ls))
      finishFileWrite(ls);
    closefd();
  }

  fileName_ = name;

  if (!writtenJSDs_) {
    writeJsds();
    writtenJSDs_ = true;
  }

  outfd_ = open(fileName_.c_str(), O_WRONLY | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
  edm::LogInfo("RawEventFileWriterForBU") << " opened " << fileName_;

  if (outfd_ < 0) {  //attention here... it may happen that outfd_ is *not* set (e.g. missing initialize call...)
    throw cms::Exception("RawEventFileWriterForBU", "initialize")
        << "Error opening FED Raw Data event output file: " << name << ": " << strerror(errno) << "\n";
  }

  perFileEventCount_.value() = 0;
  perFileSize_.value() = 0;

  adlera_ = 1;
  adlerb_ = 0;

  if (frdFileVersion_ == 1) {
    //reserve space for file header
    ftruncate(outfd_, sizeof(FRDFileHeader_v1));
    lseek(outfd_, sizeof(FRDFileHeader_v1), SEEK_SET);
    perFileSize_.value() = sizeof(FRDFileHeader_v1);
  } else if (frdFileVersion_ == 2) {
    ftruncate(outfd_, sizeof(FRDFileHeader_v2));
    lseek(outfd_, sizeof(FRDFileHeader_v2), SEEK_SET);
    perFileSize_.value() = sizeof(FRDFileHeader_v2);
  }
  assert(frdFileVersion_ <= 2);
}

void RawEventFileWriterForBU::writeJsds() {
  std::stringstream ss;
  ss << destinationDir_ << "/jsd";
  mkdir(ss.str().c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

  std::string rawJSDName = ss.str() + "/rawData.jsd";
  std::string eolJSDName = ss.str() + "/EoLS.jsd";
  std::string eorJSDName = ss.str() + "/EoR.jsd";

  fileMon_->setDefPath(rawJSDName);
  lumiMon_->setDefPath(eolJSDName);
  runMon_->setDefPath(eorJSDName);

  struct stat fstat;
  if (stat(rawJSDName.c_str(), &fstat) != 0) {
    std::string content;
    JSONSerializer::serialize(&rawJsonDef_, content);
    FileIO::writeStringToFile(rawJSDName, content);
  }

  if (stat(eolJSDName.c_str(), &fstat) != 0) {
    std::string content;
    JSONSerializer::serialize(&eolJsonDef_, content);
    FileIO::writeStringToFile(eolJSDName, content);
  }

  if (stat(eorJSDName.c_str(), &fstat) != 0) {
    std::string content;
    JSONSerializer::serialize(&eorJsonDef_, content);
    FileIO::writeStringToFile(eorJSDName, content);
  }
}

void RawEventFileWriterForBU::finishFileWrite(unsigned int ls) {
  if (frdFileVersion_ == 1) {
    //rewind
    lseek(outfd_, 0, SEEK_SET);
    FRDFileHeader_v1 frdFileHeader(perFileEventCount_.value(), (uint32_t)ls, perFileSize_.value());
    write(outfd_, (char*)&frdFileHeader, sizeof(FRDFileHeader_v1));
    closefd();
    //move raw file from open to run directory
    rename(fileName_.c_str(), (destinationDir_ + fileName_.substr(fileName_.rfind('/'))).c_str());

    edm::LogInfo("RawEventFileWriterForBU")
        << "Wrote RAW input file: " << fileName_ << " with perFileEventCount = " << perFileEventCount_.value()
        << " and size " << perFileSize_.value();
  } else if (frdFileVersion_ == 2) {
    lseek(outfd_, 0, SEEK_SET);
    FRDFileHeader_v2 frdFileHeader(0, perFileEventCount_.value(), (uint32_t)run_, (uint32_t)ls, perFileSize_.value());
    write(outfd_, (char*)&frdFileHeader, sizeof(FRDFileHeader_v2));
    closefd();
    //move raw file from open to run directory
    rename(fileName_.c_str(), (destinationDir_ + fileName_.substr(fileName_.rfind('/'))).c_str());
    edm::LogInfo("RawEventFileWriterForBU")
        << "Wrote RAW input file: " << fileName_ << " with perFileEventCount = " << perFileEventCount_.value()
        << " and size " << perFileSize_.value();
  } else {
    closefd();
    //move raw file from open to run directory
    rename(fileName_.c_str(), (destinationDir_ + fileName_.substr(fileName_.rfind('/'))).c_str());
    //create equivalent JSON file
    //TODO:fix this to use DaqDirector convention and better extension replace
    std::filesystem::path source(fileName_);
    std::string path = source.replace_extension(".jsn").string();

    fileMon_->snap(ls);
    fileMon_->outputFullJSON(path, ls);
    fileMon_->discardCollected(ls);

    //move the json file from open
    rename(path.c_str(), (destinationDir_ + path.substr(path.rfind('/'))).c_str());

    edm::LogInfo("RawEventFileWriterForBU")
        << "Wrote JSON input file: " << path << " with perFileEventCount = " << perFileEventCount_.value()
        << " and size " << perFileSize_.value();
  }
  //there is a small chance that script gets interrupted while this isn't consistent (non-atomic)
  perLumiFileCount_.value()++;
  perLumiEventCount_.value() += perFileEventCount_.value();
  perLumiSize_.value() += perFileSize_.value();
  perLumiTotalEventCount_.value() += perFileEventCount_.value();
  //update open lumi value when first file is completed
  lumiOpen_ = ls;
}

void RawEventFileWriterForBU::endOfLS(unsigned int ls) {
  if (outfd_ != -1) {
    finishFileWrite(ls);
    closefd();
  }
  lumiMon_->snap(ls);

  std::ostringstream ostr;

  ostr << destinationDir_ << "/" << runPrefix_ << "_ls" << std::setfill('0') << std::setw(4) << ls << "_EoLS"
       << ".jsn";
  //outfd_ = open(ostr.str().c_str(), O_WRONLY | O_CREAT,  S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);
  //closefd();

  std::string path = ostr.str();
  lumiMon_->outputFullJSON(path, ls);
  lumiMon_->discardCollected(ls);

  perRunEventCount_.value() += perLumiEventCount_.value();
  perRunTotalEventCount_.value() = perRunEventCount_.value();
  perRunFileCount_.value() += perLumiFileCount_.value();
  perRunLumiCount_.value() += 1;
  perRunLastLumi_.value() = ls;

  perLumiEventCount_ = 0;
  perLumiFileCount_ = 0;
  perLumiTotalEventCount_ = 0;
  perLumiSize_ = 0;
  lumiClosed_ = ls;
}

void RawEventFileWriterForBU::stop() {
  if (lumiOpen_ > lumiClosed_)
    endOfLS(lumiOpen_);
  edm::LogInfo("RawEventFileWriterForBU") << "Writing EOR file!";
  if (!destinationDir_.empty()) {
    // create EoR file
    std::string path = destinationDir_ + "/" + runPrefix_ + "_ls0000_EoR.jsn";
    runMon_->snap(0);
    runMon_->outputFullJSON(path, 0);
  }
}

void RawEventFileWriterForBU::extendDescription(edm::ParameterSetDescription& desc) {
  desc.add<int>("microSleep", 0);
  desc.add<unsigned int>("frdFileVersion", 0);
}

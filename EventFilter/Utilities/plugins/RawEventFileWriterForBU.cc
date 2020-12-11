#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
//#include <boost/filesystem/fstream.hpp>

// CMSSW headers
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"
#include "IOPool/Streamer/interface/FRDFileHeader.h"

using namespace jsoncollector;

//TODO:get run directory information from DaqDirector

RawEventFileWriterForBU::RawEventFileWriterForBU(edm::ParameterSet const& ps)
    : microSleep_(ps.getParameter<int>("microSleep")),
      frdFileVersion_(ps.getParameter<unsigned int>("frdFileVersion")) {
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

  perRunEventCount_.setName("NEvents");
  perRunFileCount_.setName("NFiles");
  perRunLumiCount_.setName("NLumis");
  perRunLastLumi_.setName("LastLumi");

  runMon_ = new FastMonitor(&eorJsonDef_, false);
  runMon_->registerGlobalMonitorable(&perRunEventCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunFileCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunLumiCount_, false, nullptr);
  runMon_->registerGlobalMonitorable(&perRunLastLumi_, false, nullptr);
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

void RawEventFileWriterForBU::initialize(std::string const& destinationDir, std::string const& name, int ls) {
  destinationDir_ = destinationDir;

  if (outfd_ != -1) {
    finishFileWrite(ls);
    closefd();
  }

  fileName_ = name;

  if (!writtenJSDs_) {
    writeJsds();
    /*    std::stringstream ss;
    ss << destinationDir_ << "/jsd";
    mkdir(ss.str().c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    std::string rawJSDName = ss.str()+"/rawData.jsd";
    std::string eolJSDName = ss.str()+"/EoLS.jsd";
    std::string eorJSDName = ss.str()+"/EoR.jsd";

    fileMon_->setDefPath(rawJSDName);
    lumiMon_->setDefPath(eolJSDName);
    runMon_->setDefPath(eorJSDName);

    struct stat   fstat;
    if (stat (rawJSDName.c_str(), &fstat) != 0) {
      std::string content;
      JSONSerializer::serialize(&rawJsonDef_,content);
      FileIO::writeStringToFile(rawJSDName, content);
    }

    if (stat (eolJSDName.c_str(), &fstat) != 0) {
      std::string content;
      JSONSerializer::serialize(&eolJsonDef_,content);
      FileIO::writeStringToFile(eolJSDName, content);
    }

    if (stat (eorJSDName.c_str(), &fstat) != 0) {
      std::string content;
      JSONSerializer::serialize(&eorJsonDef_,content);
      FileIO::writeStringToFile(eorJSDName, content);
    }
*/
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

  if (frdFileVersion_ > 0) {
    assert(frdFileVersion_ == 1);
    //reserve space for file header
    ftruncate(outfd_, sizeof(FRDFileHeader_v1));
    lseek(outfd_, sizeof(FRDFileHeader_v1), SEEK_SET);
    perFileSize_.value() = sizeof(FRDFileHeader_v1);
  }
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

void RawEventFileWriterForBU::finishFileWrite(int ls) {
  if (frdFileVersion_ > 0) {
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

void RawEventFileWriterForBU::endOfLS(int ls) {
  if (outfd_ != -1) {
    finishFileWrite(ls);
    closefd();
  }
  lumiMon_->snap(ls);

  std::ostringstream ostr;

  if (run_ == -1)
    makeRunPrefix(destinationDir_);

  ostr << destinationDir_ << "/" << runPrefix_ << "_ls" << std::setfill('0') << std::setw(4) << ls << "_EoLS"
       << ".jsn";
  //outfd_ = open(ostr.str().c_str(), O_WRONLY | O_CREAT,  S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);
  //closefd();

  std::string path = ostr.str();
  lumiMon_->outputFullJSON(path, ls);
  lumiMon_->discardCollected(ls);

  perRunEventCount_.value() += perLumiEventCount_.value();
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
    if (run_ == -1)
      makeRunPrefix(destinationDir_);
    std::string path = destinationDir_ + "/" + runPrefix_ + "_ls0000_EoR.jsn";
    runMon_->snap(0);
    runMon_->outputFullJSON(path, 0);
  }
}

//TODO:get from DaqDirector !
void RawEventFileWriterForBU::makeRunPrefix(std::string const& destinationDir) {
  //dirty hack: extract run number from destination directory
  std::string::size_type pos = destinationDir.find("run");
  std::string run = destinationDir.substr(pos + 3);
  run_ = atoi(run.c_str());
  std::stringstream ss;
  ss << "run" << std::setfill('0') << std::setw(6) << run_;
  runPrefix_ = ss.str();
}

void RawEventFileWriterForBU::extendDescription(edm::ParameterSetDescription& desc) {
  desc.add<int>("microSleep", 0);
  desc.add<unsigned int>("frdFileVersion", 0);
}

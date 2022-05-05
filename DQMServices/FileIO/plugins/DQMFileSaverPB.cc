#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <boost/property_tree/json_parser.hpp>
#include <openssl/md5.h>
#include <fmt/printf.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/gzip_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <TString.h>
#include <TSystem.h>
#include <TBufferFile.h>

#include "zlib.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/ROOTFilePB.pb.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMFileSaverPB.h"

using namespace dqm;

DQMFileSaverPB::DQMFileSaverPB(const edm::ParameterSet& ps) : DQMFileSaverBase(ps) {
  fakeFilterUnitMode_ = ps.getUntrackedParameter<bool>("fakeFilterUnitMode", false);
  streamLabel_ = ps.getUntrackedParameter<std::string>("streamLabel", "streamDQMHistograms");
  tag_ = ps.getUntrackedParameter<std::string>("tag", "UNKNOWN");

  transferDestination_ = "";
  mergeType_ = "";

  // If tag is set we're running in a DQM Live mode.
  // Snapshot files will be saved for every client, then they will be merged and uploaded to the new DQM GUI.
  if (tag_ != "UNKNOWN") {
    streamLabel_ = "DQMLive";
  }
}

DQMFileSaverPB::~DQMFileSaverPB() = default;

void DQMFileSaverPB::initRun() const {
  if (!fakeFilterUnitMode_) {
    transferDestination_ = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(streamLabel_);
    mergeType_ = edm::Service<evf::EvFDaqDirector>()->getStreamMergeType(streamLabel_, evf::MergeTypePB);
  }

  if (!fakeFilterUnitMode_) {
    evf::EvFDaqDirector* daqDirector = (evf::EvFDaqDirector*)(edm::Service<evf::EvFDaqDirector>().operator->());
    const std::string initFileName = daqDirector->getInitFilePath(streamLabel_);
    std::ofstream file(initFileName);
    file.close();
  }
}

void DQMFileSaverPB::saveLumi(const FileParameters& fp) const {
  // get from DAQ2 services where to store the files according to their format
  namespace bpt = boost::property_tree;

  std::string openJsonFilePathName;
  std::string jsonFilePathName;
  std::string openHistoFilePathName;
  std::string histoFilePathName;

  evf::FastMonitoringService* fms = nullptr;
  edm::Service<DQMStore> store;

  // create the files names
  if (fakeFilterUnitMode_) {
    std::string runDir = fmt::sprintf("%s/run%06d", fp.path_, fp.run_);
    std::string baseName = "";
    std::filesystem::create_directories(runDir);
    // If tag is configured, append it to the name of the resulting file.
    // This differentiates files saved by different clients.
    // If tag is not configured, we don't add it at all to keep the old behaviour unchanged.
    if (tag_ == "UNKNOWN") {
      baseName = fmt::sprintf("%s/run%06d_ls%04d_%s", runDir, fp.run_, fp.lumi_, streamLabel_);
    } else {
      baseName = fmt::sprintf("%s/run%06d_%s_%s", runDir, fp.run_, tag_, streamLabel_);
    }

    jsonFilePathName = baseName + ".jsn";
    openJsonFilePathName = jsonFilePathName + ".open";

    histoFilePathName = baseName + ".pb";
    openHistoFilePathName = histoFilePathName + ".open";
  } else {
    openJsonFilePathName = edm::Service<evf::EvFDaqDirector>()->getOpenOutputJsonFilePath(fp.lumi_, streamLabel_);
    jsonFilePathName = edm::Service<evf::EvFDaqDirector>()->getOutputJsonFilePath(fp.lumi_, streamLabel_);

    openHistoFilePathName =
        edm::Service<evf::EvFDaqDirector>()->getOpenProtocolBufferHistogramFilePath(fp.lumi_, streamLabel_);
    histoFilePathName = edm::Service<evf::EvFDaqDirector>()->getProtocolBufferHistogramFilePath(fp.lumi_, streamLabel_);

    fms = (evf::FastMonitoringService*)(edm::Service<evf::MicroStateService>().operator->());
  }

  bool abortFlag = false;
  if (fms ? fms->getEventsProcessedForLumi(fp.lumi_, &abortFlag) : true) {
    // Save the file in the open directory.
    this->savePB(&*store, openHistoFilePathName, fp.run_, fp.lumi_);

    // Now move the the data and json files into the output directory.
    ::rename(openHistoFilePathName.c_str(), histoFilePathName.c_str());
  }

  if (abortFlag)
    return;

  // Write the json file in the open directory.
  bpt::ptree pt = fillJson(fp.run_, fp.lumi_, histoFilePathName, transferDestination_, mergeType_, fms);
  write_json(openJsonFilePathName, pt);
  ::rename(openJsonFilePathName.c_str(), jsonFilePathName.c_str());
}

void DQMFileSaverPB::saveRun(const FileParameters& fp) const {
  // no saving for the run
}

boost::property_tree::ptree DQMFileSaverPB::fillJson(int run,
                                                     int lumi,
                                                     const std::string& dataFilePathName,
                                                     const std::string& transferDestinationStr,
                                                     const std::string& mergeTypeStr,
                                                     evf::FastMonitoringService* fms) {
  namespace bpt = boost::property_tree;
  namespace bfs = std::filesystem;

  bpt::ptree pt;

  int hostnameReturn;
  char host[32];
  hostnameReturn = gethostname(host, sizeof(host));
  if (hostnameReturn == -1)
    throw cms::Exception("fillJson") << "Internal error, cannot get host name";

  int pid = getpid();
  std::ostringstream oss_pid;
  oss_pid << pid;

  int nProcessed = fms ? (fms->getEventsProcessedForLumi(lumi)) : -1;

  // Stat the data file: if not there, throw
  std::string dataFileName;
  struct stat dataFileStat;
  dataFileStat.st_size = 0;
  if (nProcessed) {
    if (stat(dataFilePathName.c_str(), &dataFileStat) != 0)
      throw cms::Exception("fillJson") << "Internal error, cannot get data file: " << dataFilePathName;
    // Extract only the data file name from the full path
    dataFileName = bfs::path(dataFilePathName).filename().string();
  }
  // The availability test of the FastMonitoringService was done in the ctor.
  bpt::ptree data;
  bpt::ptree processedEvents, acceptedEvents, errorEvents, bitmask, fileList, fileSize, inputFiles, fileAdler32,
      transferDestination, mergeType, hltErrorEvents;

  processedEvents.put("", nProcessed);  // Processed events
  acceptedEvents.put("", nProcessed);   // Accepted events, same as processed for our purposes

  errorEvents.put("", 0);                               // Error events
  bitmask.put("", 0);                                   // Bitmask of abs of CMSSW return code
  fileList.put("", dataFileName);                       // Data file the information refers to
  fileSize.put("", dataFileStat.st_size);               // Size in bytes of the data file
  inputFiles.put("", "");                               // We do not care about input files!
  fileAdler32.put("", -1);                              // placeholder to match output json definition
  transferDestination.put("", transferDestinationStr);  // SM Transfer destination field
  mergeType.put("", mergeTypeStr);                      // SM Transfer destination field
  hltErrorEvents.put("", 0);                            // Error events

  data.push_back(std::make_pair("", processedEvents));
  data.push_back(std::make_pair("", acceptedEvents));
  data.push_back(std::make_pair("", errorEvents));
  data.push_back(std::make_pair("", bitmask));
  data.push_back(std::make_pair("", fileList));
  data.push_back(std::make_pair("", fileSize));
  data.push_back(std::make_pair("", inputFiles));
  data.push_back(std::make_pair("", fileAdler32));
  data.push_back(std::make_pair("", transferDestination));
  data.push_back(std::make_pair("", mergeType));
  data.push_back(std::make_pair("", hltErrorEvents));

  pt.add_child("data", data);

  if (fms == nullptr) {
    pt.put("definition", "/fakeDefinition.jsn");
  } else {
    // The availability test of the EvFDaqDirector Service was done in the ctor.
    bfs::path outJsonDefName{
        edm::Service<evf::EvFDaqDirector>()->baseRunDir()};  //we assume this file is written bu the EvF Output module
    outJsonDefName /= (std::string("output_") + oss_pid.str() + std::string(".jsd"));
    pt.put("definition", outJsonDefName.string());
  }

  char sourceInfo[64];  //host and pid information
  sprintf(sourceInfo, "%s_%d", host, pid);
  pt.put("source", sourceInfo);

  return pt;
}

void DQMFileSaverPB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Saves histograms from DQM store, HLT->pb workflow.");

  desc.addUntracked<bool>("fakeFilterUnitMode", false)->setComment("If set, EvFDaqDirector is emulated and not used.");

  desc.addUntracked<std::string>("streamLabel", "streamDQMHistograms")->setComment("Label of the stream.");

  DQMFileSaverBase::fillDescription(desc);

  // Changed to use addDefault instead of add here because previously
  // DQMFileSaverOnline and DQMFileSaverPB both used the module label
  // "saver" which caused conflicting cfi filenames to be generated.
  // add could be used if unique module labels were given.
  descriptions.addDefault(desc);
}

void DQMFileSaverPB::savePB(DQMStore* store, std::string const& filename, int run, int lumi) const {
  using google::protobuf::io::FileOutputStream;
  using google::protobuf::io::GzipOutputStream;
  using google::protobuf::io::StringOutputStream;

  unsigned int nme = 0;

  dqmstorepb::ROOTFilePB dqmstore_message;

  // We save all histograms, indifferent of the lumi flag: even tough we save per lumi, this is a *snapshot*.
  auto mes = store->getAllContents("");
  for (auto const me : mes) {
    TBufferFile buffer(TBufferFile::kWrite);
    if (me->kind() < MonitorElement::Kind::TH1F) {
      TObjString object(me->tagString().c_str());
      buffer.WriteObject(&object);
    } else {
      buffer.WriteObject(me->getRootObject());
    }
    dqmstorepb::ROOTFilePB::Histo& histo = *dqmstore_message.add_histo();
    histo.set_full_pathname(me->getFullname());
    uint32_t flags = 0;
    flags |= (uint32_t)me->kind();
    if (me->getLumiFlag())
      flags |= DQMNet::DQM_PROP_LUMI;
    if (me->getEfficiencyFlag())
      flags |= DQMNet::DQM_PROP_EFFICIENCY_PLOT;
    histo.set_flags(flags);
    histo.set_size(buffer.Length());

    if (tag_ == "UNKNOWN") {
      histo.set_streamed_histo((void const*)buffer.Buffer(), buffer.Length());
    } else {
      // Compress ME blob with zlib
      int maxOutputSize = this->getMaxCompressedSize(buffer.Length());
      char compression_output[maxOutputSize];
      uLong total_out = this->compressME(buffer, maxOutputSize, compression_output);
      histo.set_streamed_histo(compression_output, total_out);
    }

    // Save quality reports
    for (const auto& qr : me->getQReports()) {
      std::string result;
      // TODO: 64 is likely too short; memory corruption in the old code?
      char buf[64];
      std::snprintf(buf, sizeof(buf), "qr=st:%d:%.*g:", qr->getStatus(), DBL_DIG + 2, qr->getQTresult());
      result = '<' + me->getName() + '.' + qr->getQRName() + '>';
      result += buf;
      result += qr->getAlgorithm() + ':' + qr->getMessage();
      result += "</" + me->getName() + '.' + qr->getQRName() + '>';
      TObjString str(result.c_str());

      dqmstorepb::ROOTFilePB::Histo& qr_histo = *dqmstore_message.add_histo();
      TBufferFile qr_buffer(TBufferFile::kWrite);
      qr_buffer.WriteObject(&str);
      qr_histo.set_full_pathname(me->getFullname() + '.' + qr->getQRName());
      qr_histo.set_flags(static_cast<uint32_t>(MonitorElement::Kind::STRING));
      qr_histo.set_size(qr_buffer.Length());
      // qr_histo.set_streamed_histo((void const*)qr_buffer.Buffer(), qr_buffer.Length());

      if (tag_ == "UNKNOWN") {
        qr_histo.set_streamed_histo((void const*)qr_buffer.Buffer(), qr_buffer.Length());
      } else {
        // Compress ME blob with zlib
        int maxOutputSize = this->getMaxCompressedSize(qr_buffer.Length());
        char compression_output[maxOutputSize];
        uLong total_out = this->compressME(qr_buffer, maxOutputSize, compression_output);
        qr_histo.set_streamed_histo(compression_output, total_out);
      }
    }

    // Save efficiency tag, if any.
    // XXX not supported by protobuf files.

    // Save tag if any.
    // XXX not supported by protobuf files.

    // Count saved histograms
    ++nme;
  }

  int filedescriptor =
      ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
  FileOutputStream file_stream(filedescriptor);
  if (tag_ == "UNKNOWN") {
    GzipOutputStream::Options options;
    options.format = GzipOutputStream::GZIP;
    options.compression_level = 1;
    GzipOutputStream gzip_stream(&file_stream, options);
    dqmstore_message.SerializeToZeroCopyStream(&gzip_stream);

    // Flush the internal streams & Close the file descriptor
    gzip_stream.Close();
    file_stream.Close();
  } else {
    // We zlib compressed individual MEs so no need to compress the entire file again.
    dqmstore_message.SerializeToZeroCopyStream(&file_stream);

    // Flush the internal stream & Close the file descriptor
    file_stream.Close();
  }

  // Maybe make some noise.
  edm::LogInfo("DQMFileSaverPB") << "savePB: successfully wrote " << nme << " objects  "
                                 << "into DQM file '" << filename << "'\n";
}

int DQMFileSaverPB::getMaxCompressedSize(int bufferSize) const {
  // When input data is very badly compressable, zlib will add overhead instead of reducing the size.
  // There is a minor amount of overhead (6 bytes overall and 5 bytes per 16K block) that is taken
  // into consideration here to find out potential absolute maximum size of the output.
  int n16kBlocks = (bufferSize + 16383) / 16384;  // round up any fraction of a block
  int maxOutputSize = bufferSize + 6 + (n16kBlocks * 5);
  return maxOutputSize;
}

ulong DQMFileSaverPB::compressME(const TBufferFile& buffer, int maxOutputSize, char* compression_output) const {
  z_stream deflateStream;
  deflateStream.zalloc = Z_NULL;
  deflateStream.zfree = Z_NULL;
  deflateStream.opaque = Z_NULL;
  deflateStream.avail_in = (uInt)buffer.Length() + 1;   // size of input, string + terminator
  deflateStream.next_in = (Bytef*)buffer.Buffer();      // input array
  deflateStream.avail_out = (uInt)maxOutputSize;        // size of output
  deflateStream.next_out = (Bytef*)compression_output;  // output array, result will be placed here

  // The actual compression
  deflateInit(&deflateStream, Z_BEST_COMPRESSION);
  deflate(&deflateStream, Z_FINISH);
  deflateEnd(&deflateStream);

  return deflateStream.total_out;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMFileSaverPB);

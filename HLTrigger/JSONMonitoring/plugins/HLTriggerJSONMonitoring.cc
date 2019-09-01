/** \class HLTriggerJSONMonitoring
 *
 *
 *  Description: This class outputs JSON files with HLT monitoring information.
 *
 */

#include <atomic>
#include <fstream>

#include <boost/format.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

struct HLTriggerJSONMonitoringData {
  // variables accumulated event by event in each stream
  struct stream {
    unsigned int processed;               // number of events processed
    std::vector<unsigned int> hltWasRun;  // number of events where each path was run
    std::vector<unsigned int> hltL1s;     // number of events where each path passed the L1 seed
    std::vector<unsigned int> hltPre;     // number of events where each path passed the prescale
    std::vector<unsigned int> hltAccept;  // number of events accepted by each path
    std::vector<unsigned int> hltReject;  // number of events rejected by each path
    std::vector<unsigned int> hltErrors;  // number of events with errors in each path
    std::vector<unsigned int> datasets;   // number of events accepted by each dataset
  };

  // variables initialised for each run
  struct run {
    std::string streamDestination;
    std::string streamMergeType;
    std::string baseRunDir;   // base directory from EvFDaqDirector
    std::string jsdFileName;  // definition file name for JSON with rates

    HLTConfigProvider hltConfig;  // HLT configuration for the current run
    std::vector<int> posL1s;      // position of last L1T HLT seed filter in each path, or -1 if not present
    std::vector<int> posPre;      // position of last HLT prescale filter in each path, or -1 if not present
    std::vector<std::vector<unsigned int>> datasets;  // list of paths in each dataset
  };

  // variables accumulated over the whole lumisection
  struct lumisection {
    jsoncollector::HistoJ<unsigned int> processed;  // number of events processed
    jsoncollector::HistoJ<unsigned int> hltWasRun;  // number of events where each path was run
    jsoncollector::HistoJ<unsigned int> hltL1s;     // number of events where each path passed the L1 seed
    jsoncollector::HistoJ<unsigned int> hltPre;     // number of events where each path passed the prescale
    jsoncollector::HistoJ<unsigned int> hltAccept;  // number of events accepted by each path
    jsoncollector::HistoJ<unsigned int> hltReject;  // number of events rejected by each path
    jsoncollector::HistoJ<unsigned int> hltErrors;  // number of events with errors in each path
    jsoncollector::HistoJ<unsigned int> datasets;   // number of events accepted by each dataset
  };
};

class HLTriggerJSONMonitoring : public edm::global::EDAnalyzer<
                                    // per-stream information
                                    edm::StreamCache<HLTriggerJSONMonitoringData::stream>,
                                    // per-run accounting
                                    edm::RunCache<HLTriggerJSONMonitoringData::run>,
                                    // accumulate per-lumisection statistics
                                    edm::LuminosityBlockSummaryCache<HLTriggerJSONMonitoringData::lumisection>> {
public:
  // constructor
  explicit HLTriggerJSONMonitoring(const edm::ParameterSet&);

  // destructor
  ~HLTriggerJSONMonitoring() override = default;

  // validate the configuration and optionally fill the default values
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // called for each Event
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  // -- inherited from edm::StreamCache<HLTriggerJSONMonitoringData::stream>

  // called once for each Stream being used in the job to create the cache object that will be used for that particular Stream
  std::unique_ptr<HLTriggerJSONMonitoringData::stream> beginStream(edm::StreamID) const override;

  // called when the Stream is switching from one LuminosityBlock to a new LuminosityBlock.
  void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override;

  // -- inherited from edm::RunCache<HLTriggerJSONMonitoringData::run>

  // called each time the Source sees a new Run, and guaranteed to finish before any Stream calls streamBeginRun for that same Run
  std::shared_ptr<HLTriggerJSONMonitoringData::run> globalBeginRun(edm::Run const&,
                                                                   edm::EventSetup const&) const override;

  // called after all Streams have finished processing a given Run (i.e. streamEndRun for all Streams have completed)
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override;

  // -- inherited from edm::LuminosityBlockSummaryCache<HLTriggerJSONMonitoringData::lumisection>

  // called each time the Source sees a new LuminosityBlock
  std::shared_ptr<HLTriggerJSONMonitoringData::lumisection> globalBeginLuminosityBlockSummary(
      edm::LuminosityBlock const&, edm::EventSetup const&) const override;

  // called when a Stream has finished processing a LuminosityBlock, after streamEndLuminosityBlock
  void streamEndLuminosityBlockSummary(edm::StreamID,
                                       edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       HLTriggerJSONMonitoringData::lumisection*) const override;

  // called after the streamEndLuminosityBlockSummary method for all Streams have finished processing a given LuminosityBlock
  void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       HLTriggerJSONMonitoringData::lumisection*) const override;

private:
  static constexpr const char* streamName_ = "streamHLTRates";

  static void writeJsdFile(HLTriggerJSONMonitoringData::run const&);
  static void writeIniFile(HLTriggerJSONMonitoringData::run const&, unsigned int);

  // configuration
  const edm::InputTag triggerResults_;                               // InputTag for TriggerResults
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;  // Token for TriggerResults
};

// constructor
HLTriggerJSONMonitoring::HLTriggerJSONMonitoring(edm::ParameterSet const& config)
    : triggerResults_(config.getParameter<edm::InputTag>("triggerResults")),
      triggerResultsToken_(consumes<edm::TriggerResults>(triggerResults_)) {}

// validate the configuration and optionally fill the default values
void HLTriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  descriptions.add("HLTriggerJSONMonitoring", desc);
}

// called once for each Stream being used in the job to create the cache object that will be used for that particular Stream
std::unique_ptr<HLTriggerJSONMonitoringData::stream> HLTriggerJSONMonitoring::beginStream(edm::StreamID) const {
  return std::make_unique<HLTriggerJSONMonitoringData::stream>();
}

// called each time the Source sees a new Run, and guaranteed to finish before any Stream calls streamBeginRun for that same Run
std::shared_ptr<HLTriggerJSONMonitoringData::run> HLTriggerJSONMonitoring::globalBeginRun(
    edm::Run const& run, edm::EventSetup const& setup) const {
  auto rundata = std::make_shared<HLTriggerJSONMonitoringData::run>();

  // set the DAQ parameters
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) {
    rundata->streamDestination = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations(streamName_);
    rundata->streamMergeType =
        edm::Service<evf::EvFDaqDirector>()->getStreamMergeType(streamName_, evf::MergeTypeJSNDATA);
    rundata->baseRunDir = edm::Service<evf::EvFDaqDirector>()->baseRunDir();
  } else {
    rundata->streamDestination = "";
    rundata->streamMergeType = "";
    rundata->baseRunDir = ".";
  }

  // initialize HLTConfigProvider
  bool changed = true;
  if (not rundata->hltConfig.init(run, setup, triggerResults_.process(), changed)) {
    edm::LogError("HLTriggerJSONMonitoring") << "HLTConfigProvider initialization failed!" << std::endl;
  } else if (changed) {
    // update the trigger and dataset names
    auto const& triggerNames = rundata->hltConfig.triggerNames();
    auto const& datasetNames = rundata->hltConfig.datasetNames();
    auto const& datasets = rundata->hltConfig.datasetContents();

    const unsigned int triggersSize = triggerNames.size();
    const unsigned int datasetsSize = datasetNames.size();

    // extract the definition of the datasets
    rundata->datasets.resize(datasetsSize);
    for (unsigned int ds = 0; ds < datasetsSize; ++ds) {
      auto& dataset = rundata->datasets[ds];
      unsigned int paths = datasets[ds].size();
      dataset.reserve(paths);
      for (unsigned int p = 0; p < paths; p++) {
        unsigned int index = rundata->hltConfig.triggerIndex(datasets[ds][p]);
        if (index < triggersSize)
          dataset.push_back(index);
      }
    }

    // find the positions of the L1 seed and prescale filters
    rundata->posL1s.resize(triggersSize);
    rundata->posPre.resize(triggersSize);
    for (unsigned int i = 0; i < triggersSize; ++i) {
      rundata->posL1s[i] = -1;
      rundata->posPre[i] = -1;
      std::vector<std::string> const& moduleLabels = rundata->hltConfig.moduleLabels(i);
      for (unsigned int j = 0; j < moduleLabels.size(); ++j) {
        std::string const& label = rundata->hltConfig.moduleType(moduleLabels[j]);
        if (label == "HLTL1TSeed")
          rundata->posL1s[i] = j;
        else if (label == "HLTPrescaler")
          rundata->posPre[i] = j;
      }
    }
  }

  // write the per-run .jsd file
  rundata->jsdFileName = (boost::format("run%06d_ls0000_streamHLTRates_pid%05d.jsd") % run.run() % getpid()).str();
  writeJsdFile(*rundata);

  // write the per-run .ini file
  // iniFileName = (boost::format("run%06d_ls0000_streamHLTRates_pid%05d.ini") % run.run() % getpid()).str();
  writeIniFile(*rundata, run.run());

  return rundata;
}

// called after all Streams have finished processing a given Run (i.e. streamEndRun for all Streams have completed)
void HLTriggerJSONMonitoring::globalEndRun(edm::Run const&, edm::EventSetup const&) const {}

// called for each Event
void HLTriggerJSONMonitoring::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const {
  auto& stream = *streamCache(sid);
  auto const& rundata = *runCache(event.getRun().index());

  ++stream.processed;

  // check that the HLTConfigProvider for the current run has been successfully initialised
  if (not rundata.hltConfig.inited())
    return;

  // get hold of TriggerResults
  edm::Handle<edm::TriggerResults> handle;
  if (not event.getByToken(triggerResultsToken_, handle) or not handle.isValid()) {
    edm::LogError("HLTriggerJSONMonitoring")
        << "TriggerResults with label [" + triggerResults_.encode() + "] not present or invalid";
    return;
  }
  edm::TriggerResults const& results = *handle;
  assert(results.size() == stream.hltWasRun.size());

  // check the results for each HLT path
  for (unsigned int i = 0; i < results.size(); ++i) {
    auto const& status = results.at(i);
    if (status.wasrun()) {
      ++stream.hltWasRun[i];
      if (status.accept()) {
        ++stream.hltL1s[i];
        ++stream.hltPre[i];
        ++stream.hltAccept[i];
      } else {
        int index = (int)status.index();
        if (index > rundata.posL1s[i])
          ++stream.hltL1s[i];
        if (index > rundata.posPre[i])
          ++stream.hltPre[i];
        if (status.error())
          ++stream.hltErrors[i];
        else
          ++stream.hltReject[i];
      }
    }
  }

  // check the decision for each dataset
  // FIXME this ignores the prescales, "smart" prescales, and event selection applied in the OutputModule itself
  for (unsigned int i = 0; i < rundata.datasets.size(); ++i)
    if (std::any_of(rundata.datasets[i].begin(), rundata.datasets[i].end(), [&](unsigned int path) {
          return results.accept(path);
        }))
      ++stream.datasets[i];
}

// called each time the Source sees a new LuminosityBlock
std::shared_ptr<HLTriggerJSONMonitoringData::lumisection> HLTriggerJSONMonitoring::globalBeginLuminosityBlockSummary(
    edm::LuminosityBlock const& lumi, edm::EventSetup const&) const {
  unsigned int triggers = 0;
  unsigned int datasets = 0;
  auto const& rundata = *runCache(lumi.getRun().index());
  if (rundata.hltConfig.inited()) {
    triggers = rundata.hltConfig.triggerNames().size();
    datasets = rundata.hltConfig.datasetNames().size();
  };

  // the API of jsoncollector::HistoJ does not really match our use case,
  // but it is the only vector-like object available in JsonMonitorable.h
  auto lumidata = std::make_shared<HLTriggerJSONMonitoringData::lumisection>(HLTriggerJSONMonitoringData::lumisection{
      jsoncollector::HistoJ<unsigned int>(1),         // processed
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltWasRun
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltL1s
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltPre
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltAccept
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltReject
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltErrors
      jsoncollector::HistoJ<unsigned int>(datasets)   // datasets
  });
  // repeated calls to `update` necessary to set the internal element counter
  lumidata->processed.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltWasRun.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltL1s.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltPre.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltAccept.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltReject.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltErrors.update(0);
  for (unsigned int i = 0; i < datasets; ++i)
    lumidata->datasets.update(0);

  return lumidata;
}

// called when the Stream is switching from one LuminosityBlock to a new LuminosityBlock.
void HLTriggerJSONMonitoring::streamBeginLuminosityBlock(edm::StreamID sid,
                                                         edm::LuminosityBlock const& lumi,
                                                         edm::EventSetup const&) const {
  auto& stream = *streamCache(sid);

  unsigned int triggers = 0;
  unsigned int datasets = 0;
  auto const& rundata = *runCache(lumi.getRun().index());
  if (rundata.hltConfig.inited()) {
    triggers = rundata.hltConfig.triggerNames().size();
    datasets = rundata.hltConfig.datasetNames().size();
  };

  // reset the stream counters
  stream.processed = 0;
  stream.hltWasRun.assign(triggers, 0);
  stream.hltL1s.assign(triggers, 0);
  stream.hltPre.assign(triggers, 0);
  stream.hltAccept.assign(triggers, 0);
  stream.hltReject.assign(triggers, 0);
  stream.hltErrors.assign(triggers, 0);
  stream.datasets.assign(datasets, 0);
}

// called when a Stream has finished processing a LuminosityBlock, after streamEndLuminosityBlock
void HLTriggerJSONMonitoring::streamEndLuminosityBlockSummary(edm::StreamID sid,
                                                              edm::LuminosityBlock const& lumi,
                                                              edm::EventSetup const&,
                                                              HLTriggerJSONMonitoringData::lumisection* lumidata) const {
  auto const& stream = *streamCache(sid);
  auto const& rundata = *runCache(lumi.getRun().index());
  lumidata->processed.value()[0] += stream.processed;

  // check that the HLTConfigProvider for the current run has been successfully initialised
  if (not rundata.hltConfig.inited())
    return;

  unsigned int triggers = rundata.hltConfig.triggerNames().size();
  for (unsigned int i = 0; i < triggers; ++i) {
    lumidata->hltWasRun.value()[i] += stream.hltWasRun[i];
    lumidata->hltL1s.value()[i] += stream.hltL1s[i];
    lumidata->hltPre.value()[i] += stream.hltPre[i];
    lumidata->hltAccept.value()[i] += stream.hltAccept[i];
    lumidata->hltReject.value()[i] += stream.hltReject[i];
    lumidata->hltErrors.value()[i] += stream.hltErrors[i];
  }
  unsigned int datasets = rundata.hltConfig.datasetNames().size();
  for (unsigned int i = 0; i < datasets; ++i)
    lumidata->datasets.value()[i] += stream.datasets[i];
}

// called after the streamEndLuminosityBlockSummary method for all Streams have finished processing a given LuminosityBlock
void HLTriggerJSONMonitoring::globalEndLuminosityBlockSummary(edm::LuminosityBlock const& lumi,
                                                              edm::EventSetup const&,
                                                              HLTriggerJSONMonitoringData::lumisection* lumidata) const {
  unsigned int ls = lumi.luminosityBlock();
  unsigned int run = lumi.run();

  bool writeFiles = true;
  if (edm::Service<evf::MicroStateService>().isAvailable()) {
    evf::FastMonitoringService* fms =
        (evf::FastMonitoringService*)(edm::Service<evf::MicroStateService>().operator->());
    if (fms)
      writeFiles = fms->shouldWriteFiles(ls);
  }
  if (not writeFiles)
    return;

  unsigned int processed = lumidata->processed.value().at(0);
  auto const& rundata = *runCache(lumi.getRun().index());
  Json::StyledWriter writer;

  // [SIC]
  char hostname[33];
  gethostname(hostname, 32);
  std::string sourceHost(hostname);

  // [SIC]
  std::stringstream sOutDef;
  sOutDef << rundata.baseRunDir << "/"
          << "output_" << getpid() << ".jsd";

  std::string jsndataFileList = "";
  unsigned int jsndataSize = 0;
  unsigned int jsndataAdler32 = 1;  // adler32 checksum for an empty file

  if (processed) {
    // write the .jsndata files which contain the actual rates
    Json::Value jsndata;
    jsndata[jsoncollector::DataPoint::SOURCE] = sourceHost;
    jsndata[jsoncollector::DataPoint::DEFINITION] = rundata.jsdFileName;
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->processed.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltWasRun.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltL1s.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltPre.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltAccept.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltReject.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltErrors.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->datasets.toJsonValue());

    auto jsndataFileName = boost::format("run%06d_ls%04d_streamHLTRates_pid%05d.jsndata") % run % ls % getpid();

    std::string result = writer.write(jsndata);
    std::ofstream jsndataFile(rundata.baseRunDir + "/" + jsndataFileName.str());
    jsndataFile << result;
    jsndataFile.close();

    jsndataFileList = jsndataFileName.str();
    jsndataSize = result.size();
    jsndataAdler32 = cms::Adler32(result.c_str(), result.size());
  }

  // create a metadata json file for the "HLT rates" pseudo-stream
  unsigned int jsnProcessed = processed;
  unsigned int jsnAccepted = processed;
  unsigned int jsnErrorEvents = 0;
  unsigned int jsnRetCodeMask = 0;
  std::string jsnInputFiles = "";
  unsigned int jsnHLTErrorEvents = 0;

  Json::Value jsn;
  jsn[jsoncollector::DataPoint::SOURCE] = sourceHost;
  jsn[jsoncollector::DataPoint::DEFINITION] = sOutDef.str();
  jsn[jsoncollector::DataPoint::DATA].append(jsnProcessed);
  jsn[jsoncollector::DataPoint::DATA].append(jsnAccepted);
  jsn[jsoncollector::DataPoint::DATA].append(jsnErrorEvents);
  jsn[jsoncollector::DataPoint::DATA].append(jsnRetCodeMask);
  jsn[jsoncollector::DataPoint::DATA].append(jsndataFileList);
  jsn[jsoncollector::DataPoint::DATA].append(jsndataSize);
  jsn[jsoncollector::DataPoint::DATA].append(jsnInputFiles);
  jsn[jsoncollector::DataPoint::DATA].append(jsndataAdler32);
  jsn[jsoncollector::DataPoint::DATA].append(rundata.streamDestination);
  jsn[jsoncollector::DataPoint::DATA].append(rundata.streamMergeType);
  jsn[jsoncollector::DataPoint::DATA].append(jsnHLTErrorEvents);

  auto jsnFileName = boost::format("run%06d_ls%04d_streamHLTRates_pid%05d.jsn") % run % ls % getpid();
  std::ofstream jsnFile(rundata.baseRunDir + "/" + jsnFileName.str());
  jsnFile << writer.write(jsn);
  jsnFile.close();
}

void HLTriggerJSONMonitoring::writeJsdFile(HLTriggerJSONMonitoringData::run const& rundata) {
  std::ofstream file(rundata.baseRunDir + "/" + rundata.jsdFileName);
  file << R"""({
   "data" : [
      { "name" : "Processed", "type" : "integer", "operation" : "histo"},
      { "name" : "Path-WasRun", "type" : "integer", "operation" : "histo"},
      { "name" : "Path-AfterL1Seed", "type" : "integer", "operation" : "histo"},
      { "name" : "Path-AfterPrescale", "type" : "integer", "operation" : "histo"},
      { "name" : "Path-Accepted", "type" : "integer", "operation" : "histo"},
      { "name" : "Path-Rejected", "type" : "integer", "operation" : "histo"},
      { "name" : "Path-Errors", "type" : "integer", "operation" : "histo"},
      { "name" : "Dataset-Accepted", "type" : "integer", "operation" : "histo"}
   ]
}
)""";
  file.close();
}

void HLTriggerJSONMonitoring::writeIniFile(HLTriggerJSONMonitoringData::run const& rundata, unsigned int run) {
  Json::Value content;

  Json::Value triggerNames(Json::arrayValue);
  for (auto const& name : rundata.hltConfig.triggerNames())
    triggerNames.append(name);
  content["Path-Names"] = triggerNames;

  Json::Value datasetNames(Json::arrayValue);
  for (auto const& name : rundata.hltConfig.datasetNames())
    datasetNames.append(name);
  content["Dataset-Names"] = datasetNames;

  std::string iniFileName = (boost::format("run%06d_ls0000_streamHLTRates_pid%05d.ini") % run % getpid()).str();
  std::ofstream file(rundata.baseRunDir + "/" + iniFileName);
  Json::StyledWriter writer;
  file << writer.write(content);
  file.close();
}

// declare as a framework plugin
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTriggerJSONMonitoring);

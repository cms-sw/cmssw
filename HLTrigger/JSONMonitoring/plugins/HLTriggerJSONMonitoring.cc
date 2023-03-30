/** \class HLTriggerJSONMonitoring
 *
 *  Description: This class outputs JSON files with HLT monitoring information.
 *
 */

#include <atomic>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include <fmt/printf.h>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//note this was updated 20/10/22 to change the logic such that instead
//of having it passing the L1Seed, it is now number passing pre prescale
//this is indentical for "standard" paths and more meaningful for
//the special paths which are affected
//a standard path logic goes "trigger type -> l1 seed -> prescale -> other selection"

struct HLTriggerJSONMonitoringData {
  // variables accumulated event by event in each stream
  struct stream {
    unsigned int processed;               // number of events processed
    std::vector<unsigned int> hltWasRun;  // number of events where each path was run
    std::vector<unsigned int> hltPrePS;   // number of events where each path made it to the prescale module
    std::vector<unsigned int> hltPostPS;  // number of events where each path passed the prescale
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
    std::vector<int> posPre;      // position of last HLT prescale filter in each path, or -1 if not present
    std::vector<std::vector<unsigned int>> datasets;  // list of paths in each dataset
    std::vector<unsigned int> indicesOfTriggerPaths;  // indices of triggers (without DatasetPaths) in TriggerNames
  };

  // variables accumulated over the whole lumisection
  struct lumisection {
    jsoncollector::HistoJ<unsigned int> processed;  // number of events processed
    jsoncollector::HistoJ<unsigned int> hltWasRun;  // number of events where each path was run
    jsoncollector::HistoJ<unsigned int> hltPrePS;   // number of events where each path made it to the prescale module
    jsoncollector::HistoJ<unsigned int> hltPostPS;  // number of events where each path passed the prescale
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

  static constexpr const char* datasetPathNamePrefix_ = "Dataset_";

  static void writeJsdFile(HLTriggerJSONMonitoringData::run const&);
  static void writeIniFile(HLTriggerJSONMonitoringData::run const&, unsigned int);

  // configuration
  const edm::InputTag triggerResults_;                               // InputTag for TriggerResults
  const edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;  // Token for TriggerResults
};

// constructor
HLTriggerJSONMonitoring::HLTriggerJSONMonitoring(edm::ParameterSet const& config)
    : triggerResults_(config.getParameter<edm::InputTag>("triggerResults")),
      triggerResultsToken_(consumes(triggerResults_)) {
  if (edm::Service<evf::EvFDaqDirector>().isAvailable()) {
    //output initemp file. This lets hltd know number of streams early
    std::string initFileName = edm::Service<evf::EvFDaqDirector>()->getInitTempFilePath("streamHLTRates");
    std::ofstream file(initFileName);
    if (!file)
      throw cms::Exception("HLTriggerJsonMonitoring")
          << "Cannot create INITEMP file: " << initFileName << " error: " << strerror(errno);
    file.close();
  }
}

// validate the configuration and optionally fill the default values
void HLTriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "@currentProcess"));
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
    edm::LogError("HLTriggerJSONMonitoring") << "HLTConfigProvider initialization failed!";
  } else if (changed) {
    // triggerNames from TriggerResults (includes DatasetPaths)
    auto const& triggerNames = rundata->hltConfig.triggerNames();
    auto const triggerNamesSize = triggerNames.size();

    // update the list of indices of the HLT Paths (without DatasetPaths) in the TriggerNames list
    rundata->indicesOfTriggerPaths.clear();
    rundata->indicesOfTriggerPaths.reserve(triggerNamesSize);
    for (auto triggerNameIdx = 0u; triggerNameIdx < triggerNamesSize; ++triggerNameIdx) {
      // skip DatasetPaths
      if (triggerNames[triggerNameIdx].find(datasetPathNamePrefix_) != 0) {
        rundata->indicesOfTriggerPaths.emplace_back(triggerNameIdx);
      }
    }
    auto const triggersSize = rundata->indicesOfTriggerPaths.size();

    // update the list of paths in each dataset
    auto const& datasets = rundata->hltConfig.datasetContents();
    auto const& datasetNames = rundata->hltConfig.datasetNames();
    auto const datasetsSize = datasetNames.size();
    rundata->datasets.resize(datasetsSize);
    for (auto ds = 0u; ds < datasetsSize; ++ds) {
      auto& dataset = rundata->datasets[ds];
      // check if TriggerNames include the DatasetPath corresponding to this Dataset
      //  - DatasetPaths are normal cms.Path objects
      //  - in Run-3 HLT menus, DatasetPaths are used to define PrimaryDatasets
      auto const datasetPathName = datasetPathNamePrefix_ + datasetNames[ds];
      auto const datasetPathExists =
          std::find(triggerNames.begin(), triggerNames.end(), datasetPathName) != triggerNames.end();
      if (datasetPathExists) {
        // if a DatasetPath exists, only that Path is assigned to the Dataset
        //  - this way, the counts of the Dataset properly include prescales on the DatasetPath
        //    and smart-Prescales applied by the DatasetPath to its triggers
        dataset.reserve(1);
        auto const index = rundata->hltConfig.triggerIndex(datasetPathName);
        if (index < triggerNamesSize)
          dataset.push_back(index);
      } else {
        auto const paths = datasets[ds].size();
        dataset.reserve(paths);
        for (auto p = 0u; p < paths; p++) {
          auto const index = rundata->hltConfig.triggerIndex(datasets[ds][p]);
          if (index < triggerNamesSize)
            dataset.push_back(index);
        }
      }
    }

    // find the positions of the prescale filters
    rundata->posPre.resize(triggersSize);
    for (auto i = 0u; i < triggersSize; ++i) {
      rundata->posPre[i] = -1;
      auto const trigNameIndx = rundata->indicesOfTriggerPaths[i];
      auto const& moduleLabels = rundata->hltConfig.moduleLabels(trigNameIndx);
      for (auto j = 0u; j < moduleLabels.size(); ++j) {
        auto const& moduleType = rundata->hltConfig.moduleType(moduleLabels[j]);
        if (moduleType == "HLTPrescaler") {
          rundata->posPre[i] = j;
          break;
        }
      }
    }
  }

  // write the per-run .jsd file
  rundata->jsdFileName = fmt::sprintf("run%06d_ls0000_streamHLTRates_pid%05d.jsd", run.run(), getpid());
  writeJsdFile(*rundata);

  // write the per-run .ini file
  //iniFileName = fmt::sprintf("run%06d_ls0000_streamHLTRates_pid%05d.ini", run.run(), getpid());
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
  assert(results.size() == rundata.hltConfig.triggerNames().size());

  // check the results for each HLT path
  for (auto idx = 0u; idx < rundata.indicesOfTriggerPaths.size(); ++idx) {
    auto const triggerPathIdx = rundata.indicesOfTriggerPaths[idx];
    auto const& status = results[triggerPathIdx];
    if (status.wasrun()) {
      ++stream.hltWasRun[idx];
      if (status.accept()) {
        ++stream.hltPrePS[idx];
        ++stream.hltPostPS[idx];
        ++stream.hltAccept[idx];
      } else {
        int const index = (int)status.index();
        if (index >= rundata.posPre[idx])
          ++stream.hltPrePS[idx];
        if (index > rundata.posPre[idx])
          ++stream.hltPostPS[idx];
        if (status.error())
          ++stream.hltErrors[idx];
        else
          ++stream.hltReject[idx];
      }
    }
  }

  // check the decision for each dataset
  // FIXME this ignores the prescales, "smart" prescales, and event selection applied in the OutputModule itself
  for (auto i = 0u; i < rundata.datasets.size(); ++i)
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
    triggers = rundata.indicesOfTriggerPaths.size();
    datasets = rundata.hltConfig.datasetNames().size();
  };

  // the API of jsoncollector::HistoJ does not really match our use case,
  // but it is the only vector-like object available in JsonMonitorable.h
  auto lumidata = std::make_shared<HLTriggerJSONMonitoringData::lumisection>(HLTriggerJSONMonitoringData::lumisection{
      jsoncollector::HistoJ<unsigned int>(1),         // processed
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltWasRun
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltPrePS
      jsoncollector::HistoJ<unsigned int>(triggers),  // hltPostPS
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
    lumidata->hltPrePS.update(0);
  for (unsigned int i = 0; i < triggers; ++i)
    lumidata->hltPostPS.update(0);
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
    triggers = rundata.indicesOfTriggerPaths.size();
    datasets = rundata.hltConfig.datasetNames().size();
  };

  // reset the stream counters
  stream.processed = 0;
  stream.hltWasRun.assign(triggers, 0);
  stream.hltPrePS.assign(triggers, 0);
  stream.hltPostPS.assign(triggers, 0);
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

  auto const triggers = rundata.indicesOfTriggerPaths.size();
  for (auto i = 0u; i < triggers; ++i) {
    lumidata->hltWasRun.value()[i] += stream.hltWasRun[i];
    lumidata->hltPrePS.value()[i] += stream.hltPrePS[i];
    lumidata->hltPostPS.value()[i] += stream.hltPostPS[i];
    lumidata->hltAccept.value()[i] += stream.hltAccept[i];
    lumidata->hltReject.value()[i] += stream.hltReject[i];
    lumidata->hltErrors.value()[i] += stream.hltErrors[i];
  }
  auto const datasets = rundata.hltConfig.datasetNames().size();
  for (auto i = 0u; i < datasets; ++i)
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
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltPrePS.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltPostPS.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltAccept.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltReject.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->hltErrors.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->datasets.toJsonValue());

    auto jsndataFileName = fmt::sprintf("run%06d_ls%04d_streamHLTRates_pid%05d.jsndata", run, ls, getpid());

    std::string result = writer.write(jsndata);
    std::ofstream jsndataFile(rundata.baseRunDir + "/" + jsndataFileName);
    jsndataFile << result;
    jsndataFile.close();

    jsndataFileList = jsndataFileName;
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

  auto jsnFileName = fmt::sprintf("run%06d_ls%04d_streamHLTRates_pid%05d.jsn", run, ls, getpid());
  std::ofstream jsnFile(rundata.baseRunDir + "/" + jsnFileName);
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
  for (auto idx : rundata.indicesOfTriggerPaths)
    triggerNames.append(rundata.hltConfig.triggerNames()[idx]);
  content["Path-Names"] = triggerNames;

  Json::Value datasetNames(Json::arrayValue);
  for (auto const& name : rundata.hltConfig.datasetNames())
    datasetNames.append(name);
  content["Dataset-Names"] = datasetNames;

  std::string iniFileName = fmt::sprintf("run%06d_ls0000_streamHLTRates_pid%05d.ini", run, getpid());
  std::ofstream file(rundata.baseRunDir + "/" + iniFileName);
  Json::StyledWriter writer;
  file << writer.write(content);
  file.close();
}

// declare as a framework plugin
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTriggerJSONMonitoring);

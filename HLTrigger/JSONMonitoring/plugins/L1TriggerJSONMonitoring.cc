/** \class L1TriggerJSONMonitoring
 *
 *  Description: This class outputs JSON files with TCDS and L1T monitoring information.
 *
 */

#include <atomic>
#include <fstream>

#include <boost/format.hpp>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
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
#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"

struct L1TriggerJSONMonitoringData {
  // special values for prescale index checks
  static constexpr const int kPrescaleUndefined = -2;
  static constexpr const int kPrescaleConflict = -1;
  // variables accumulated event by event in each stream
  struct stream {
    unsigned int processed = 0;                      // number of events processed
    std::vector<unsigned int> l1tAccept;             // number of events accepted by each L1 trigger
    std::vector<unsigned int> l1tAcceptPhysics;      // number of "physics" events accepted by each L1 trigger
    std::vector<unsigned int> l1tAcceptCalibration;  // number of "calibration" events accepted by each L1 trigger
    std::vector<unsigned int> l1tAcceptRandom;       // number of "random" events accepted by each L1 trigger
    std::vector<unsigned int> tcdsAccept;    // number of "physics", "calibration", "random" and other event types
    int prescaleIndex = kPrescaleUndefined;  // prescale column index
  };

  // variables initialised for each run
  struct run {
    std::string streamDestination;
    std::string streamMergeType;
    std::string baseRunDir;   // base directory from EvFDaqDirector
    std::string jsdFileName;  // definition file name for JSON with rates
  };

  // variables accumulated over the whole lumisection
  struct lumisection {
    jsoncollector::HistoJ<unsigned int> processed;         // number of events processed
    jsoncollector::HistoJ<unsigned int> l1tAccept;         // number of events accepted by each L1 trigger
    jsoncollector::HistoJ<unsigned int> l1tAcceptPhysics;  // number of "physics" events accepted by each L1 trigger
    jsoncollector::HistoJ<unsigned int>
        l1tAcceptCalibration;                             // number of "calibration" events accepted by each L1 trigger
    jsoncollector::HistoJ<unsigned int> l1tAcceptRandom;  // number of "random" events accepted by each L1 trigger
    jsoncollector::HistoJ<unsigned int> tcdsAccept;  // number of "physics", "calibration", "random" and other event types
    int prescaleIndex = kPrescaleUndefined;          // prescale column index
  };
};

class L1TriggerJSONMonitoring : public edm::global::EDAnalyzer<
                                    // per-stream information
                                    edm::StreamCache<L1TriggerJSONMonitoringData::stream>,
                                    // per-run accounting
                                    edm::RunCache<L1TriggerJSONMonitoringData::run>,
                                    // accumulate per-lumisection statistics
                                    edm::LuminosityBlockSummaryCache<L1TriggerJSONMonitoringData::lumisection> > {
public:
  // constructor
  explicit L1TriggerJSONMonitoring(const edm::ParameterSet&);

  // destructor
  ~L1TriggerJSONMonitoring() override = default;

  // validate the configuration and optionally fill the default values
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // called for each Event
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  // -- inherited from edm::StreamCache<L1TriggerJSONMonitoringData::stream>

  // called once for each Stream being used in the job to create the cache object that will be used for that particular Stream
  std::unique_ptr<L1TriggerJSONMonitoringData::stream> beginStream(edm::StreamID) const override;

  // called when the Stream is switching from one LuminosityBlock to a new LuminosityBlock.
  void streamBeginLuminosityBlock(edm::StreamID, edm::LuminosityBlock const&, edm::EventSetup const&) const override;

  // -- inherited from edm::RunCache<L1TriggerJSONMonitoringData::run>

  // called each time the Source sees a new Run, and guaranteed to finish before any Stream calls streamBeginRun for that same Run
  std::shared_ptr<L1TriggerJSONMonitoringData::run> globalBeginRun(edm::Run const&,
                                                                   edm::EventSetup const&) const override;

  // called after all Streams have finished processing a given Run (i.e. streamEndRun for all Streams have completed)
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override;

  // -- inherited from edm::LuminosityBlockSummaryCache<L1TriggerJSONMonitoringData::lumisection>

  // called each time the Source sees a new LuminosityBlock
  std::shared_ptr<L1TriggerJSONMonitoringData::lumisection> globalBeginLuminosityBlockSummary(
      edm::LuminosityBlock const&, edm::EventSetup const&) const override;

  // called when a Stream has finished processing a LuminosityBlock, after streamEndLuminosityBlock
  void streamEndLuminosityBlockSummary(edm::StreamID,
                                       edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       L1TriggerJSONMonitoringData::lumisection*) const override;

  // called after the streamEndLuminosityBlockSummary method for all Streams have finished processing a given LuminosityBlock
  void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                       edm::EventSetup const&,
                                       L1TriggerJSONMonitoringData::lumisection*) const override;

private:
  // TCDS trigger types
  // see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TcdsEventRecord
  static constexpr const std::array<const char*, 16> tcdsTriggerTypes_ = {{
      "Error",          //  0 - No trigger (DAQ error stream events may have this value)
      "Physics",        //  1 - GT trigger
      "Calibration",    //  2 - Sequence trigger (calibration)
      "Random",         //  3 - Random trigger
      "Auxiliary",      //  4 - Auxiliary (CPM front panel NIM input) trigger
      "",               //  5 - reserved
      "",               //  6 - reserved
      "",               //  7 - reserved
      "Cyclic",         //  8 - Cyclic trigger
      "Bunch-pattern",  //  9 - Bunch-pattern trigger
      "Software",       // 10 - Software trigger
      "TTS",            // 11 - TTS-sourced trigger
      "",               // 12 - reserved
      "",               // 13 - reserved
      "",               // 14 - reserved
      ""                // 15 - reserved
  }};

  static constexpr const char* streamName_ = "streamL1Rates";

  static void writeJsdFile(L1TriggerJSONMonitoringData::run const&);
  static void writeIniFile(L1TriggerJSONMonitoringData::run const&, unsigned int, std::vector<std::string> const&);

  // configuration
  const edm::InputTag level1Results_;                                    // InputTag for L1 trigge results
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> level1ResultsToken_;  // Token for L1 trigge results
};

// instantiate static data members
constexpr const std::array<const char*, 16> L1TriggerJSONMonitoring::tcdsTriggerTypes_;

// constructor
L1TriggerJSONMonitoring::L1TriggerJSONMonitoring(edm::ParameterSet const& config)
    : level1Results_(config.getParameter<edm::InputTag>("L1Results")),
      level1ResultsToken_(consumes<GlobalAlgBlkBxCollection>(level1Results_)) {}

// validate the configuration and optionally fill the default values
void L1TriggerJSONMonitoring::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1Results", edm::InputTag("hltGtStage2Digis"));
  descriptions.add("L1TriggerJSONMonitoring", desc);
}

// called once for each Stream being used in the job to create the cache object that will be used for that particular Stream
std::unique_ptr<L1TriggerJSONMonitoringData::stream> L1TriggerJSONMonitoring::beginStream(edm::StreamID) const {
  return std::make_unique<L1TriggerJSONMonitoringData::stream>();
}

// called each time the Source sees a new Run, and guaranteed to finish before any Stream calls streamBeginRun for that same Run
std::shared_ptr<L1TriggerJSONMonitoringData::run> L1TriggerJSONMonitoring::globalBeginRun(
    edm::Run const& run, edm::EventSetup const& setup) const {
  auto rundata = std::make_shared<L1TriggerJSONMonitoringData::run>();

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

  // read the L1 trigger names from the EventSetup
  std::vector<std::string> triggerNames(GlobalAlgBlk::maxPhysicsTriggers, ""s);
  edm::ESHandle<L1TUtmTriggerMenu> menuHandle;
  setup.get<L1TUtmTriggerMenuRcd>().get(menuHandle);
  if (menuHandle.isValid()) {
    for (auto const& algo : menuHandle->getAlgorithmMap())
      triggerNames[algo.second.getIndex()] = algo.first;
  } else {
    edm::LogWarning("L1TriggerJSONMonitoring") << "L1TUtmTriggerMenu not found in the EventSetup.\nThe Level 1 Trigger "
                                                  "rate monitoring will not include the trigger names.";
  }

  // write the per-run .jsd file
  rundata->jsdFileName = (boost::format("run%06d_ls0000_streamL1Rates_pid%05d.jsd") % run.run() % getpid()).str();
  writeJsdFile(*rundata);

  // write the per-run .ini file
  // iniFileName = (boost::format("run%06d_ls0000_streamL1Rates_pid%05d.ini") % run.run() % getpid()).str();
  writeIniFile(*rundata, run.run(), triggerNames);

  return rundata;
}

// called after all Streams have finished processing a given Run (i.e. streamEndRun for all Streams have completed)
void L1TriggerJSONMonitoring::globalEndRun(edm::Run const&, edm::EventSetup const&) const {}

// called for each Event
void L1TriggerJSONMonitoring::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const {
  auto& stream = *streamCache(sid);

  ++stream.processed;
  unsigned int eventType = event.experimentType();
  if (eventType < tcdsTriggerTypes_.size())
    ++stream.tcdsAccept[eventType];
  else
    edm::LogWarning("L1TriggerJSONMonitoring") << "Unexpected event type " << eventType;

  // get hold of TriggerResults
  edm::Handle<GlobalAlgBlkBxCollection> handle;
  if (not event.getByToken(level1ResultsToken_, handle) or not handle.isValid() or handle->isEmpty(0)) {
    edm::LogError("L1TriggerJSONMonitoring")
        << "L1 trigger results with label [" + level1Results_.encode() + "] not present or invalid";
    return;
  }

  // The GlobalAlgBlkBxCollection is a vector of vectors, but the second layer can only ever
  // have one entry since there can't be more than one collection per bunch crossing.
  // The first "0" here means BX = 0, while the second "0" is used to access the first and only element.
  auto const& results = handle->at(0, 0);
  auto const& decision = results.getAlgoDecisionFinal();
  assert(decision.size() == GlobalAlgBlk::maxPhysicsTriggers);

  // check the results for each HLT path
  for (unsigned int i = 0; i < decision.size(); ++i) {
    if (decision[i]) {
      ++stream.l1tAccept[i];
      switch (eventType) {
        case edm::EventAuxiliary::PhysicsTrigger:
          ++stream.l1tAcceptPhysics[i];
          break;
        case edm::EventAuxiliary::CalibrationTrigger:
          ++stream.l1tAcceptCalibration[i];
          break;
        case edm::EventAuxiliary::RandomTrigger:
          ++stream.l1tAcceptRandom[i];
          break;
        default:
          break;
      }
    }
  }

  // check for conflicting values in the prescale column index, and store it
  int prescaleIndex = results.getPreScColumn();
  if (stream.prescaleIndex == L1TriggerJSONMonitoringData::kPrescaleUndefined) {
    stream.prescaleIndex = prescaleIndex;
  } else if (stream.prescaleIndex == L1TriggerJSONMonitoringData::kPrescaleConflict) {
    // do nothing
  } else if (stream.prescaleIndex != prescaleIndex) {
    edm::LogWarning("L1TriggerJSONMonitoring") << "Prescale index changed from " << stream.prescaleIndex << " to "
                                               << prescaleIndex << " inside lumisection " << event.luminosityBlock();
    stream.prescaleIndex = L1TriggerJSONMonitoringData::kPrescaleConflict;
  }
}

// called each time the Source sees a new LuminosityBlock
std::shared_ptr<L1TriggerJSONMonitoringData::lumisection> L1TriggerJSONMonitoring::globalBeginLuminosityBlockSummary(
    edm::LuminosityBlock const& lumi, edm::EventSetup const&) const {
  // the API of jsoncollector::HistoJ does not really match our use case,
  // but it is the only vector-like object available in JsonMonitorable.h
  auto lumidata = std::make_shared<L1TriggerJSONMonitoringData::lumisection>(L1TriggerJSONMonitoringData::lumisection{
      jsoncollector::HistoJ<unsigned int>(1),                                 // processed
      jsoncollector::HistoJ<unsigned int>(GlobalAlgBlk::maxPhysicsTriggers),  // l1tAccept
      jsoncollector::HistoJ<unsigned int>(GlobalAlgBlk::maxPhysicsTriggers),  // l1tAcceptPhysics
      jsoncollector::HistoJ<unsigned int>(GlobalAlgBlk::maxPhysicsTriggers),  // l1tAcceptCalibration
      jsoncollector::HistoJ<unsigned int>(GlobalAlgBlk::maxPhysicsTriggers),  // l1tAcceptRandom
      jsoncollector::HistoJ<unsigned int>(tcdsTriggerTypes_.size())           // tcdsAccept
  });
  // repeated calls to `update` necessary to set the internal element counter
  lumidata->processed.update(0);
  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
    lumidata->l1tAccept.update(0);
  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
    lumidata->l1tAcceptPhysics.update(0);
  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
    lumidata->l1tAcceptCalibration.update(0);
  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
    lumidata->l1tAcceptRandom.update(0);
  for (unsigned int i = 0; i < tcdsTriggerTypes_.size(); ++i)
    lumidata->tcdsAccept.update(0);
  lumidata->prescaleIndex = L1TriggerJSONMonitoringData::kPrescaleUndefined;

  return lumidata;
}

// called when the Stream is switching from one LuminosityBlock to a new LuminosityBlock.
void L1TriggerJSONMonitoring::streamBeginLuminosityBlock(edm::StreamID sid,
                                                         edm::LuminosityBlock const& lumi,
                                                         edm::EventSetup const&) const {
  auto& stream = *streamCache(sid);

  // reset the stream counters
  stream.processed = 0;
  stream.l1tAccept.assign(GlobalAlgBlk::maxPhysicsTriggers, 0);
  stream.l1tAcceptPhysics.assign(GlobalAlgBlk::maxPhysicsTriggers, 0);
  stream.l1tAcceptCalibration.assign(GlobalAlgBlk::maxPhysicsTriggers, 0);
  stream.l1tAcceptRandom.assign(GlobalAlgBlk::maxPhysicsTriggers, 0);
  stream.tcdsAccept.assign(tcdsTriggerTypes_.size(), 0);
  stream.prescaleIndex = L1TriggerJSONMonitoringData::kPrescaleUndefined;
}

// called when a Stream has finished processing a LuminosityBlock, after streamEndLuminosityBlock
void L1TriggerJSONMonitoring::streamEndLuminosityBlockSummary(edm::StreamID sid,
                                                              edm::LuminosityBlock const& lumi,
                                                              edm::EventSetup const&,
                                                              L1TriggerJSONMonitoringData::lumisection* lumidata) const {
  auto const& stream = *streamCache(sid);
  lumidata->processed.value()[0] += stream.processed;

  for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i) {
    lumidata->l1tAccept.value()[i] += stream.l1tAccept[i];
    lumidata->l1tAcceptPhysics.value()[i] += stream.l1tAcceptPhysics[i];
    lumidata->l1tAcceptCalibration.value()[i] += stream.l1tAcceptCalibration[i];
    lumidata->l1tAcceptRandom.value()[i] += stream.l1tAcceptRandom[i];
  }
  for (unsigned int i = 0; i < tcdsTriggerTypes_.size(); ++i)
    lumidata->tcdsAccept.value()[i] += stream.tcdsAccept[i];

  // check for conflicting values in the prescale column index
  if (lumidata->prescaleIndex == L1TriggerJSONMonitoringData::kPrescaleUndefined)
    lumidata->prescaleIndex = stream.prescaleIndex;
  else if (lumidata->prescaleIndex != stream.prescaleIndex)
    lumidata->prescaleIndex = L1TriggerJSONMonitoringData::kPrescaleConflict;
}

// called after the streamEndLuminosityBlockSummary method for all Streams have finished processing a given LuminosityBlock
void L1TriggerJSONMonitoring::globalEndLuminosityBlockSummary(edm::LuminosityBlock const& lumi,
                                                              edm::EventSetup const&,
                                                              L1TriggerJSONMonitoringData::lumisection* lumidata) const {
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
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->l1tAccept.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->l1tAcceptPhysics.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->l1tAcceptCalibration.toJsonValue());
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->l1tAcceptRandom.toJsonValue());

    // write only the number of "physics", "calibration" and "random" events
    jsoncollector::HistoJ<unsigned int> tcdsAccept;
    tcdsAccept.update(lumidata->tcdsAccept.value()[edm::EventAuxiliary::PhysicsTrigger]);
    tcdsAccept.update(lumidata->tcdsAccept.value()[edm::EventAuxiliary::CalibrationTrigger]);
    tcdsAccept.update(lumidata->tcdsAccept.value()[edm::EventAuxiliary::RandomTrigger]);
    jsndata[jsoncollector::DataPoint::DATA].append(tcdsAccept.toJsonValue());
    /* FIXME send information for all event types instead of only these three
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->tcdsAccept.toJsonValue());
    */
    jsndata[jsoncollector::DataPoint::DATA].append(lumidata->prescaleIndex);

    auto jsndataFileName = boost::format("run%06d_ls%04d_streamL1Rates_pid%05d.jsndata") % run % ls % getpid();

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

  auto jsnFileName = boost::format("run%06d_ls%04d_streamL1Rates_pid%05d.jsn") % run % ls % getpid();
  std::ofstream jsnFile(rundata.baseRunDir + "/" + jsnFileName.str());
  jsnFile << writer.write(jsn);
  jsnFile.close();
}

void L1TriggerJSONMonitoring::writeJsdFile(L1TriggerJSONMonitoringData::run const& rundata) {
  std::ofstream file(rundata.baseRunDir + "/" + rundata.jsdFileName);
  file << R"""({
   "data" : [
      { "name" : "Processed", "type" : "integer", "operation" : "histo"},
      { "name" : "L1-AlgoAccepted", "type" : "integer", "operation" : "histo"},
      { "name" : "L1-AlgoAccepted-Physics", "type" : "integer", "operation" : "histo"},
      { "name" : "L1-AlgoAccepted-Calibration", "type" : "integer", "operation" : "histo"},
      { "name" : "L1-AlgoAccepted-Random", "type" : "integer", "operation" : "histo"},
      { "name" : "L1-Global", "type" : "integer", "operation" : "histo"},
      { "name" : "Prescale-Index", "type" : "integer", "operation" : "sample"}
   ]
})""";
  file.close();
}

void L1TriggerJSONMonitoring::writeIniFile(L1TriggerJSONMonitoringData::run const& rundata,
                                           unsigned int run,
                                           std::vector<std::string> const& l1TriggerNames) {
  Json::Value content;

  Json::Value triggerNames(Json::arrayValue);
  for (auto const& name : l1TriggerNames)
    triggerNames.append(name);
  content["L1-Algo-Names"] = triggerNames;

  Json::Value eventTypes(Json::arrayValue);
  eventTypes.append(tcdsTriggerTypes_[edm::EventAuxiliary::PhysicsTrigger]);
  eventTypes.append(tcdsTriggerTypes_[edm::EventAuxiliary::CalibrationTrigger]);
  eventTypes.append(tcdsTriggerTypes_[edm::EventAuxiliary::RandomTrigger]);
  /* FIXME send information for all event types instead of only these three
  for (auto const& name : tcdsTriggerTypes_)
    eventTypes.append(name);
  */
  content["Event-Type"] = eventTypes;

  std::string iniFileName = (boost::format("run%06d_ls0000_streamL1Rates_pid%05d.ini") % run % getpid()).str();
  std::ofstream file(rundata.baseRunDir + "/" + iniFileName);
  Json::StyledWriter writer;
  file << writer.write(content);
  file.close();
}

// declare as a framework plugin
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TriggerJSONMonitoring);

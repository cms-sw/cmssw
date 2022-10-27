#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSource.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/Averages.h"
#include "DQM/SiStripCommissioningSources/interface/CalibrationScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/CalibrationTask.h"
#include "DQM/SiStripCommissioningSources/interface/DaqScopeModeTask.h"
#include "DQM/SiStripCommissioningSources/interface/FastFedCablingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FedTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FineDelayTask.h"
#include "DQM/SiStripCommissioningSources/interface/LatencyTask.h"
#include "DQM/SiStripCommissioningSources/interface/NoiseTask.h"
#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedsFullNoiseTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedsOnlyTask.h"
#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ctime>
#include <iomanip>
#include <memory>
#include <sstream>

#include <sys/types.h>
#include <unistd.h>
#include <iomanip>

#include <arpa/inet.h>
#include <cstdio>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/unistd.h>
#include <cstdint>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::SiStripCommissioningSource(const edm::ParameterSet& pset)
    : dqm_(nullptr),
      fedCablingToken_(esConsumes<edm::Transition::BeginRun>()),
      fecCabling_(nullptr),
      inputModuleLabel_(pset.getParameter<std::string>("InputModuleLabel")),
      inputModuleLabelAlt_(pset.existsAs<std::string>("InputModuleLabelAlt")
                               ? pset.getParameter<std::string>("InputModuleLabelAlt")
                               : ""),
      inputModuleLabelSummary_(pset.getParameter<std::string>("SummaryInputModuleLabel")),
      inputClusterLabel_(
          pset.existsAs<std::string>("InputClusterLabel") ? pset.getParameter<std::string>("InputClusterLabel") : ""),
      filename_(pset.getUntrackedParameter<std::string>("RootFileName", sistrip::dqmSourceFileName_)),
      run_(0),
      partitionName_(pset.existsAs<std::string>("PartitionName") ? pset.getParameter<std::string>("PartitionName")
                                                                 : ""),
      time_(0),
      isSpy_(pset.existsAs<bool>("isSpy") ? pset.getParameter<bool>("isSpy") : false),
      taskConfigurable_(pset.getUntrackedParameter<std::string>("CommissioningTask", "UNDEFINED")),
      task_(sistrip::UNDEFINED_RUN_TYPE),
      tasks_(),
      cablingTasks_(),
      tasksExist_(false),
      cablingTask_(false),
      updateFreq_(pset.getUntrackedParameter<int>("HistoUpdateFreq", 1)),
      base_(""),
      view_(pset.getUntrackedParameter<std::string>("View", "Default")),
      parameters_(pset) {
  usesResource("DQMStore");
  inputModuleSummaryToken_ = consumes<SiStripEventSummary>(edm::InputTag(inputModuleLabelSummary_));
  digiVirginRawToken_ = mayConsume<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "VirginRaw"));
  digiFineDelaySelectionToken_ =
      mayConsume<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "FineDelaySelection"));
  digiReorderedToken_ = mayConsume<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "Reordered"));
  /////////////////
  if (not isSpy_)
    digiScopeModeToken_ = mayConsume<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabel_, "ScopeMode"));
  else {
    digiScopeModeToken_ =
        mayConsume<edm::DetSetVector<SiStripRawDigi> >(edm::InputTag(inputModuleLabelAlt_, "ScopeRawDigis"));
    clustersToken_ = mayConsume<edmNew::DetSetVector<SiStripCluster> >(edm::InputTag(inputClusterLabel_));
  }
  /////////////////
  LogTrace(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                         << " Constructing object...";
  tasks_.clear();
  tasks_.resize(1024, VecOfTasks(96, static_cast<CommissioningTask*>(nullptr)));

  if (task_ == sistrip::NOISE) {
    noiseToken_ = esConsumes();
  }
  if (task_ == sistrip::NOISE || task_ == sistrip::CALIBRATION_SCAN || task_ == sistrip::CALIBRATION_SCAN_DECO ||
      task_ == sistrip::CALIBRATION || task_ == sistrip::CALIBRATION_DECO) {
    pedestalToken_ = esConsumes();
  }
}

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::~SiStripCommissioningSource() {
  LogTrace(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                         << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::DQMStore* const SiStripCommissioningSource::dqm(std::string method) const {
  if (!dqm_) {
    std::stringstream ss;
    if (!method.empty()) {
      ss << "[SiStripCommissioningSource::" << method << "]" << std::endl;
    } else {
      ss << "[SiStripCommissioningSource]" << std::endl;
    }
    ss << " NULL pointer to DQMStore";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return nullptr;
  } else {
    return dqm_;
  }
}

// -----------------------------------------------------------------------------
// Retrieve DQM interface, control cabling and "control view" utility
// class, create histogram directory structure and generate "reverse"
// control cabling.
void SiStripCommissioningSource::beginRun(edm::Run const& run, const edm::EventSetup& setup) {
  LogTrace(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                         << " Configuring..." << std::endl;

  // ---------- DQM back-end interface ----------

  dqm_ = edm::Service<DQMStore>().operator->();
  edm::LogInfo(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                             << " DQMStore service: " << dqm_;
  dqm(__func__);

  // ---------- Base directory ----------

  std::stringstream dir("");
  base_ = dir.str();

  // ---------- FED and FEC cabling ----------

  const auto& fed_cabling = setup.getData(fedCablingToken_);
  fedCabling_ = const_cast<SiStripFedCabling*>(&fed_cabling);
  LogDebug(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                         << "Initialized FED cabling. Number of FEDs is " << fedCabling_->fedIds().size();
  fecCabling_ = new SiStripFecCabling(fed_cabling);
  if (fecCabling_->crates().empty()) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Empty std::vector returned by FEC cabling object!"
       << " Check if database connection failed...";
    edm::LogWarning(mlDqmSource_) << ss.str();
  }

  // ---------- Reset ----------

  tasksExist_ = false;
  task_ = sistrip::UNDEFINED_RUN_TYPE;
  cablingTask_ = false;

  remove();

  clearCablingTasks();
  clearTasks();
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::endJob() {
  LogTrace(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                         << " Halting..." << std::endl;

  // ---------- Update histograms ----------
  // Cabling task
  for (TaskMap::iterator itask = cablingTasks_.begin(); itask != cablingTasks_.end(); itask++) {
    if (itask->second) {
      itask->second->updateHistograms();
    }
  }

  if (task_ == sistrip::APV_LATENCY) {
    for (uint16_t partition = 0; partition < 4; ++partition) {
      tasks_[0][partition]->updateHistograms();
    }
  } else if (task_ == sistrip::FINE_DELAY) {
    tasks_[0][0]->updateHistograms();
  } else {
    // All tasks except cabling
    uint16_t fed_id = 0;
    uint16_t fed_ch = 0;
    auto ifed = fedCabling_->fedIds().begin();
    for (; ifed != fedCabling_->fedIds().end(); ifed++) {
      auto conns = fedCabling_->fedConnections(*ifed);
      for (auto iconn = conns.begin(); iconn != conns.end(); iconn++) {
        if (!iconn->isConnected()) {
          continue;
        }
        fed_id = iconn->fedId();
        fed_ch = iconn->fedCh();
        if (tasks_[fed_id][fed_ch]) {
          tasks_[fed_id][fed_ch]->updateHistograms();
          delete tasks_[fed_id][fed_ch];
        }
      }
    }
  }
  // ---------- Save histos to root file ----------

  // Strip filename of ".root" extension
  std::string name;
  if (filename_.find(".root", 0) == std::string::npos) {
    name = filename_;
  } else {
    name = filename_.substr(0, filename_.find(".root", 0));
  }

  // Retrieve SCRATCH directory
  std::string scratch = "SCRATCH";  //@@ remove trailing slash!!!
  std::string dir = "";
  if (std::getenv(scratch.c_str()) != nullptr) {
    dir = std::getenv(scratch.c_str());
  }

  // Add directory path
  std::stringstream ss;
  if (!dir.empty()) {
    ss << dir << "/";
  } else {
    ss << "/tmp/";
  }

  // Add filename with run number, host ip, pid and .root extension
  ss << name << "_";
  if (task_ == sistrip::DAQ_SCOPE_MODE and not partitionName_.empty())  // only for spy-runs
    ss << partitionName_ << "_";

  directory(ss, run_);
  ss << ".root";

  // Save file with appropriate filename (if run number is known)
  if (!filename_.empty()) {
    if (run_ != 0) {
      dqm()->save(ss.str());
    } else {
      edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                    << " NULL value for RunNumber! No root file saved!";
    }
  } else {
    edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                  << " NULL value for filename! No root file saved!";
  }

  LogTrace(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                         << " Saved all histograms to file \"" << ss.str() << "\"";

  // ---------- Delete histograms ----------
  // Remove all MonitorElements in "SiStrip" dir and below
  // remove();

  // Delete histogram objects
  // clearCablingTasks();
  // clearTasks();

  // ---------- Delete cabling ----------
  if (fedCabling_) {
    fedCabling_ = nullptr;
  }
  if (fecCabling_) {
    delete fecCabling_;
    fecCabling_ = nullptr;
  }
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // Retrieve commissioning information from "event summary"
  edm::Handle<SiStripEventSummary> summary;
  //  event.getByLabel( inputModuleLabelSummary_, summary );
  event.getByToken(inputModuleSummaryToken_, summary);

  // Check if EventSummary has info attached
  if ((summary->runType() == sistrip::UNDEFINED_RUN_TYPE || summary->runType() == sistrip::UNKNOWN_RUN_TYPE) &&
      summary->nullParams()) {
    edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                  << " Unknown/undefined RunType and NULL parameter values!"
                                  << " It may be that the 'trigger FED' object was not found!";
  }

  // Check if need to rebuild FED/FEC cabling objects for connection run
  //cablingForConnectionRun( summary->runType() ); //@@ do not use!

  // Extract run number and forward to client
  if (event.id().run() != run_) {
    run_ = event.id().run();
    createRunNumber();
  }

  // Coarse event rate counter
  if (!(event.id().event() % updateFreq_)) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " The last " << updateFreq_ << " events were processed at a rate of ";
    if (time(nullptr) == time_) {
      ss << ">" << updateFreq_ << " Hz";
    } else {
      ss << (updateFreq_ / (time(nullptr) - time_)) << " Hz";
    }
    edm::LogVerbatim(mlDqmSource_) << ss.str();
    time_ = time(nullptr);
  }

  // Create commissioning task objects
  if (!tasksExist_) {
    createTask(summary.product(), setup);
  } else {
    for (auto& v : tasks_) {
      for (auto& t : v) {
        t->eventSetup(&setup);
      }
    }
  }

  // Retrieve raw digis with mode appropriate to task
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > raw;
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > rawAlt;
  edm::Handle<edmNew::DetSetVector<SiStripCluster> > cluster;

  if (task_ == sistrip::DAQ_SCOPE_MODE) {  // scop-mode runs
    if (not isSpy_ and
        summary->fedReadoutMode() == FED_VIRGIN_RAW) {  // if the readout is virgin raw just take the VR digis
      event.getByToken(digiVirginRawToken_, raw);
    } else if (not isSpy_ and summary->fedReadoutMode() == FED_SCOPE_MODE) {
      event.getByToken(digiScopeModeToken_, raw);
    } else if (isSpy_) {  // spy case take both re-ordered and scope mode
      event.getByToken(digiScopeModeToken_, rawAlt);
      event.getByToken(digiReorderedToken_, raw);
      if (not inputClusterLabel_.empty())
        event.getByToken(clustersToken_, cluster);
    } else {
      std::stringstream ss;
      ss << "[SiStripCommissioningSource::" << __func__ << "]"
         << " Requested DAQ_SCOPE_MODE but unknown FED"
         << " readout mode retrieved from SiStripEventSummary: "
         << SiStripEnumsAndStrings::fedReadoutMode(summary->fedReadoutMode());
      edm::LogWarning(mlDqmSource_) << ss.str();
    }
  } else if (task_ == sistrip::FAST_CABLING || task_ == sistrip::FED_CABLING || task_ == sistrip::APV_TIMING ||
             task_ == sistrip::FED_TIMING || task_ == sistrip::OPTO_SCAN) {
    event.getByToken(digiScopeModeToken_, raw);
  } else if (task_ == sistrip::VPSP_SCAN || task_ == sistrip::CALIBRATION || task_ == sistrip::CALIBRATION_DECO ||
             task_ == sistrip::CALIBRATION_SCAN || task_ == sistrip::CALIBRATION_SCAN_DECO ||
             task_ == sistrip::PEDESTALS || task_ == sistrip::PEDS_ONLY || task_ == sistrip::NOISE ||
             task_ == sistrip::PEDS_FULL_NOISE) {
    event.getByToken(digiVirginRawToken_, raw);
  } else if (task_ == sistrip::APV_LATENCY || task_ == sistrip::FINE_DELAY) {
    event.getByToken(digiFineDelaySelectionToken_, raw);
  } else {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unknown CommissioningTask: " << SiStripEnumsAndStrings::runType(task_)
       << " Unable to establish FED readout mode and retrieve digi container!"
       << " Check if SiStripEventSummary object is found/present in Event";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }

  // Check for NULL pointer to digi container
  if (not raw.isValid()) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << std::endl
       << " NULL pointer to DetSetVector!" << std::endl
       << " Unable to fill histograms!";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }

  if (isSpy_ and not inputModuleLabelAlt_.empty() and not rawAlt.isValid()) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << std::endl
       << " NULL pointer to DetSetVector!" << std::endl
       << " Unable to fill histograms!";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }

  if (isSpy_ and not inputClusterLabel_.empty() and not cluster.isValid()) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << std::endl
       << " NULL pointer to DetSetVector!" << std::endl
       << " Unable to fill histograms!";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }

  if (!cablingTask_) {
    fillHistos(summary.product(), *raw, *rawAlt, *cluster);
  } else {
    fillCablingHistos(summary.product(), *raw);
  }
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::fillCablingHistos(const SiStripEventSummary* const summary,
                                                   const edm::DetSetVector<SiStripRawDigi>& raw) {
  // Create FEC key using DCU id and LLD channel from SiStripEventSummary
  const SiStripModule& module = fecCabling_->module(summary->dcuId());
  uint16_t lld_channel = (summary->deviceId() & 0x3) + 1;
  SiStripFecKey key_object(module.key().fecCrate(),
                           module.key().fecSlot(),
                           module.key().fecRing(),
                           module.key().ccuAddr(),
                           module.key().ccuChan(),
                           lld_channel);
  uint32_t fec_key = key_object.key();
  std::stringstream sss;
  sss << "[SiStripCommissioningSource::" << __func__ << "]"
      << " Found DcuId 0x" << std::hex << std::setw(8) << std::setfill('0') << summary->dcuId() << std::dec
      << " with Crate/FEC/Ring/CCU/Module/LLD: " << module.key().fecCrate() << "/" << module.key().fecSlot() << "/"
      << module.key().fecRing() << "/" << module.key().ccuAddr() << "/" << module.key().ccuChan() << "/" << lld_channel;
  edm::LogWarning(mlDqmSource_) << sss.str();

  //LogTrace(mlTest_) << "TEST : " << key_object;

  // Check on whether DCU id is found
  if (key_object.isInvalid(sistrip::CCU_CHAN)) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " DcuId 0x" << std::hex << std::setw(8) << std::setfill('0') << summary->dcuId() << std::dec
       << " in 'DAQ register' field not found in cabling map!"
       << " (NULL values returned for FEC path)";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }

  // Iterate through FED ids
  for (auto ifed = fedCabling_->fedIds().begin(); ifed != fedCabling_->fedIds().end(); ifed++) {
    // Check if FedId is non-zero
    if (*ifed == sistrip::invalid_) {
      continue;
    }

    // Container to hold median signal level for FED cabling task
    std::map<uint16_t, float> medians;
    medians.clear();
    std::map<uint16_t, float> medians1;
    medians1.clear();

    // Iterate through FED channels
    for (uint16_t ichan = 0; ichan < 96; ichan++) {
      // Retrieve digis for given FED key
      uint32_t fed_key = ((*ifed & sistrip::invalid_) << 16) | (ichan & sistrip::invalid_);

      std::vector<edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find(fed_key);
      if (digis != raw.end()) {
        if (digis->data.empty()) {
          continue;
        }

        Averages ave;
        for (uint16_t idigi = 0; idigi < digis->data.size(); idigi++) {
          ave.add(static_cast<uint32_t>(digis->data[idigi].adc()));
        }
        Averages::Params params;
        ave.calc(params);
        medians[ichan] = params.median_;  // Store median signal level
        medians1[ichan] = digis->data[0].adc();
      }

    }  // fed channel loop

    // Calculate mean and spread on all (median) signal levels
    Averages average;
    std::map<uint16_t, float>::const_iterator ii = medians.begin();
    for (; ii != medians.end(); ii++) {
      average.add(ii->second);
    }
    Averages::Params tmp;
    average.calc(tmp);

    std::stringstream ss;
    ss << "FED Averages:" << std::endl
       << "  nChans: " << medians.size() << std::endl
       << "  num/mean/median/rms/max/min: " << tmp.num_ << "/" << tmp.mean_ << "/" << tmp.median_ << "/" << tmp.rms_
       << "/" << tmp.max_ << "/" << tmp.min_ << std::endl;
    LogTrace(mlDqmSource_) << ss.str();

    // Calculate mean and spread on "filtered" data
    Averages truncated;
    std::map<uint16_t, float>::const_iterator jj = medians.begin();
    for (; jj != medians.end(); jj++) {
      if (jj->second < tmp.median_ + tmp.rms_) {
        truncated.add(jj->second);
      }
    }
    Averages::Params params;
    truncated.calc(params);

    std::stringstream ss1;
    ss1 << "Truncated Averages:" << std::endl
        << "  nChans: " << medians.size() << std::endl
        << "  num/mean/median/rms/max/min: " << params.num_ << "/" << params.mean_ << "/" << params.median_ << "/"
        << params.rms_ << "/" << params.max_ << "/" << params.min_ << std::endl;
    LogTrace(mlDqmSource_) << ss1.str();

    // Identify channels with signal
    std::stringstream ss2;
    std::stringstream ss3;
    std::map<uint16_t, float> channels;
    std::map<uint16_t, float>::const_iterator ichan = medians.begin();
    for (; ichan != medians.end(); ichan++) {
      if (ichan->second > 200.) {
        LogTrace(mlTest_) << "TEST FOUND SIGNAL HIGH: " << *ifed << " " << ichan->first << " " << ichan->second;
        channels[ichan->first] = ichan->second;
      }
      ss2  //<< ichan->first << "/"
          << ichan->second << " ";
      ss3  //<< ichan->first << "/"
          << medians1[ichan->first] << " ";
    }

    ss2 << std::endl;
    ss3 << std::endl;
    LogTrace(mlTest_) << "DUMP for FED  " << *ifed << ": " << ss2.str();
    LogTrace(mlTest_) << "FIRST ADC VAL " << *ifed << ": " << ss3.str();

    // Fill cabling histograms
    if (cablingTasks_.find(fec_key) != cablingTasks_.end()) {
      if (!channels.empty()) {
        cablingTasks_[fec_key]->fillHistograms(*summary, *ifed, channels);
        SiStripFecKey path(fec_key);
        std::stringstream ss;
        ss << "[SiStripCommissioningSource::" << __func__ << "]"
           << " Filled histogram for '" << cablingTasks_[fec_key]->myName() << "' object with FecKey: 0x" << std::hex
           << std::setfill('0') << std::setw(8) << fec_key << std::dec
           << " and Crate/FEC/ring/CCU/module/LLDchan: " << path.fecCrate() << "/" << path.fecSlot() << "/"
           << path.fecRing() << "/" << path.ccuAddr() << "/" << path.ccuChan() << "/" << path.channel();
        LogTrace(mlDqmSource_) << ss.str();
      }
    } else {
      SiStripFecKey path(fec_key);
      std::stringstream ss;
      ss << "[SiStripCommissioningSource::" << __func__ << "]"
         << " Unable to find CommissioningTask object with FecKey: 0x" << std::hex << std::setfill('0') << std::setw(8)
         << fec_key << std::dec << " and Crate/FEC/ring/CCU/module/LLDchan: " << path.fecCrate() << "/"
         << path.fecSlot() << "/" << path.fecRing() << "/" << path.ccuAddr() << "/" << path.ccuChan() << "/"
         << path.channel();
      edm::LogWarning(mlDqmSource_) << ss.str();
    }

  }  // fed id loop
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::fillHistos(const SiStripEventSummary* const summary,
                                            const edm::DetSetVector<SiStripRawDigi>& raw,
                                            const edm::DetSetVector<SiStripRawDigi>& rawAlt,
                                            const edmNew::DetSetVector<SiStripCluster>& clusters) {
  // Iterate through FED ids and channels
  std::vector<uint16_t> stripOnClusters;
  auto ifed = fedCabling_->fedIds().begin();
  for (; ifed != fedCabling_->fedIds().end(); ifed++) {
    // Iterate through connected FED channels
    auto conns = fedCabling_->fedConnections(*ifed);
    for (auto iconn = conns.begin(); iconn != conns.end(); iconn++) {
      if (!(iconn->fedId()) || iconn->fedId() > sistrip::valid_) {
        continue;
      }
      if (!iconn->isConnected()) {
        continue;
      }

      // Create FED key and check if non-zero
      // note: the key is not computed using the same formula as in commissioning histograms.
      // beware that changes here must match changes in raw2digi and in SiStripFineDelayHit
      uint32_t fed_key = ((iconn->fedId() & sistrip::invalid_) << 16) | (iconn->fedCh() & sistrip::invalid_);
      // Retrieve digis for given FED key and check if found
      std::vector<edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find(fed_key);

      // only for spy data-taking --> tick measurement
      std::vector<edm::DetSet<SiStripRawDigi> >::const_iterator digisAlt;
      if (not rawAlt.empty()) {
        digisAlt = rawAlt.find(fed_key);
        if (digisAlt == rawAlt.end())
          continue;
      }

      // find the strips belonging to the clusters connected to this APV pair
      stripOnClusters.clear();
      if (not clusters.empty()) {
        for (edmNew::DetSetVector<SiStripCluster>::const_iterator DSViter = clusters.begin(); DSViter != clusters.end();
             DSViter++) {
          if (DSViter->id() != iconn->detId())
            continue;  // select clusters on this module
          for (edmNew::DetSet<SiStripCluster>::const_iterator DSiter = DSViter->begin(); DSiter != DSViter->end();
               DSiter++) {  // loop on the clusters
            if (DSiter->firstStrip() >= iconn->apvPairNumber() * 256 and
                DSiter->firstStrip() < (1 + iconn->apvPairNumber()) * 256) {  // found the right APV
              for (size_t istrip = 0; istrip < DSiter->amplitudes().size(); istrip++) {
                stripOnClusters.push_back(DSiter->firstStrip() + istrip - iconn->apvPairNumber() * 256);
              }
            }
          }
        }
      }

      if (digis != raw.end()) {
        // tasks involving tracking have partition-level histos, so treat separately
        if (task_ == sistrip::APV_LATENCY) {
          if (tasks_[0][iconn->fecCrate() - 1]) {
            tasks_[0][iconn->fecCrate() - 1]->fillHistograms(*summary, *digis);
          } else {
            std::stringstream ss;
            ss << "[SiStripCommissioningSource::" << __func__ << "]"
               << " Unable to find CommissioningTask for FEC crate " << iconn->fecCrate()
               << ". Unable to fill histograms!";
            edm::LogWarning(mlDqmSource_) << ss.str();
          }
        } else if (task_ == sistrip::FINE_DELAY) {
          if (tasks_[0][0]) {
            tasks_[0][0]->fillHistograms(*summary, *digis);
          } else {
            std::stringstream ss;
            ss << "[SiStripCommissioningSource::" << __func__ << "]"
               << " Unable to find global CommissioningTask for FineDelay. Unable to fill histograms!";
            edm::LogWarning(mlDqmSource_) << ss.str();
          }
        } else {
          if (tasks_[iconn->fedId()][iconn->fedCh()]) {
            if (not rawAlt.empty() or digisAlt == rawAlt.end())
              tasks_[iconn->fedId()][iconn->fedCh()]->fillHistograms(*summary, *digis);
            else {  // for spy-data
              if (stripOnClusters.empty())
                tasks_[iconn->fedId()][iconn->fedCh()]->fillHistograms(*summary, *digis, *digisAlt);
              else {
                tasks_[iconn->fedId()][iconn->fedCh()]->fillHistograms(*summary, *digis, *digisAlt, stripOnClusters);
              }
            }
          } else {
            std::stringstream ss;
            ss << "[SiStripCommissioningSource::" << __func__ << "]"
               << " Unable to find CommissioningTask object with FED key " << std::hex << std::setfill('0')
               << std::setw(8) << fed_key << std::dec << " and FED id/ch " << iconn->fedId() << "/" << iconn->fedCh()
               << " Unable to fill histograms!";
            edm::LogWarning(mlDqmSource_) << ss.str();
          }
        }
      } else {
        // issue a warning only for standard runs, as latency and fine delay only deliver
        // pseudo zero-suppressed data
        if (task_ != sistrip::APV_LATENCY && task_ != sistrip::FINE_DELAY) {
          std::stringstream ss;
          ss << "[SiStripCommissioningSource::" << __func__ << "]"
             << " Unable to find any DetSet containing digis for FED key " << std::hex << std::setfill('0')
             << std::setw(8) << fed_key << std::dec << " and FED id/ch " << iconn->fedId() << "/" << iconn->fedCh();
          edm::LogWarning(mlDqmSource_) << ss.str();
        }
      }
    }  // fed channel loop
  }    // fed id loop
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createRunNumber() {
  // Set commissioning task to default ("undefined") value
  if (!run_) {
    edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                  << " NULL run number!";
    return;
  }

  // Create MonitorElement that identifies run number
  dqm()->setCurrentFolder(base_ + sistrip::root_);
  std::stringstream run;
  run << run_;
  dqm()->bookString(std::string(sistrip::runNumber_) + sistrip::sep_ + run.str(), run.str());
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createTask(const SiStripEventSummary* const summary, const edm::EventSetup& setup) {
  // Set commissioning task to default ("undefined") value
  task_ = sistrip::UNDEFINED_RUN_TYPE;

  // Retrieve commissioning task from EventSummary
  if (summary) {
    task_ = summary->runType();
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Identified CommissioningTask from EventSummary to be \"" << SiStripEnumsAndStrings::runType(task_) << "\"";
    LogTrace(mlDqmSource_) << ss.str();
  } else {
    task_ = sistrip::UNKNOWN_RUN_TYPE;
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " NULL pointer to SiStripEventSummary!"
       << " Check SiStripEventSummary is found/present in Event";
    edm::LogWarning(mlDqmSource_) << ss.str();
  }

  // Override task with ParameterSet configurable (if defined)
  sistrip::RunType configurable = SiStripEnumsAndStrings::runType(taskConfigurable_);
  if (configurable != sistrip::UNDEFINED_RUN_TYPE && configurable != sistrip::UNKNOWN_RUN_TYPE) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Overriding CommissioningTask from EventSummary (\"" << SiStripEnumsAndStrings::runType(task_)
       << "\") with value retrieved from .cfg file (\"" << SiStripEnumsAndStrings::runType(configurable) << "\")!";
    LogTrace(mlDqmSource_) << ss.str();
    task_ = configurable;
  }

  // Create ME (std::string) that identifies commissioning task
  dqm()->setCurrentFolder(base_ + sistrip::root_);
  std::string task_str = SiStripEnumsAndStrings::runType(task_);
  dqm()->bookString(std::string(sistrip::taskId_) + sistrip::sep_ + task_str, task_str);

  // Check commissioning task is known / defined
  if (task_ == sistrip::UNKNOWN_RUN_TYPE || task_ == sistrip::UNDEFINED_RUN_TYPE) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unexpected CommissioningTask found (" << static_cast<uint16_t>(task_) << ") \""
       << SiStripEnumsAndStrings::runType(task_) << "\""
       << " Unexpected value found in SiStripEventSummary and/or cfg file"
       << " If SiStripEventSummary is not present in Event,"
       << " check 'CommissioningTask' configurable in cfg file";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  } else {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Identified CommissioningTask to be \"" << SiStripEnumsAndStrings::runType(task_) << "\"";
    LogTrace(mlDqmSource_) << ss.str();
  }

  // Check if commissioning task is FED cabling
  if (task_ == sistrip::FED_CABLING) {
    cablingTask_ = true;
  } else {
    cablingTask_ = false;
  }

  std::stringstream ss;
  ss << "[SiStripCommissioningSource::" << __func__ << "]"
     << " CommissioningTask: " << SiStripEnumsAndStrings::runType(summary->runType());
  LogTrace(mlDqmSource_) << ss.str();

  edm::LogVerbatim(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                 << " Creating CommissioningTask objects and booking histograms...";
  if (cablingTask_) {
    createCablingTasks();
  } else {
    createTasks(task_, setup);
  }
  edm::LogVerbatim(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                 << " Finished booking histograms!";
  tasksExist_ = true;
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createCablingTasks() {
  // Iterate through FEC cabling and create commissioning task objects
  uint16_t booked = 0;
  for (std::vector<SiStripFecCrate>::const_iterator icrate = fecCabling_->crates().begin();
       icrate != fecCabling_->crates().end();
       icrate++) {
    for (std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++) {
      for (std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end();
           iring++) {
        for (std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end();
             iccu++) {
          for (std::vector<SiStripModule>::const_iterator imodule = iccu->modules().begin();
               imodule != iccu->modules().end();
               imodule++) {
            // Build FEC key
            SiStripFecKey path(
                icrate->fecCrate(), ifec->fecSlot(), iring->fecRing(), iccu->ccuAddr(), imodule->ccuChan());

            // Check if FEC key is invalid
            if (!path.isValid()) {
              continue;
            }

            // Set working directory prior to booking histograms
            std::string dir = base_ + path.path();
            dqm()->setCurrentFolder(dir);

            // Iterate through all APV pairs for this module
            for (uint16_t ipair = 0; ipair < imodule->nApvPairs(); ipair++) {
              // Retrieve active APV devices
              SiStripModule::PairOfU16 apvs = imodule->activeApvPair(imodule->lldChannel(ipair));

              // Create connection object to hold all relevant info
              FedChannelConnection conn(icrate->fecCrate(),
                                        ifec->fecSlot(),
                                        iring->fecRing(),
                                        iccu->ccuAddr(),
                                        imodule->ccuChan(),
                                        apvs.first,
                                        apvs.second,
                                        imodule->dcuId(),
                                        imodule->detId(),
                                        imodule->nApvPairs());

              // Define key encoding control path
              uint32_t key = SiStripFecKey(icrate->fecCrate(),
                                           ifec->fecSlot(),
                                           iring->fecRing(),
                                           iccu->ccuAddr(),
                                           imodule->ccuChan(),
                                           imodule->lldChannel(ipair))
                                 .key();

              // Check key is non zero
              if (!key) {
                edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                              << " Unexpected NULL value for FEC key!";
                continue;
              }

              // Create cabling task objects if not already existing
              if (cablingTasks_.find(key) == cablingTasks_.end()) {
                if (task_ == sistrip::FED_CABLING) {
                  cablingTasks_[key] = new FedCablingTask(dqm(), conn);
                } else if (task_ == sistrip::UNDEFINED_RUN_TYPE) {
                  edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                                << " Undefined CommissioningTask"
                                                << " Unable to create FedCablingTask object!";
                } else if (task_ == sistrip::UNKNOWN_RUN_TYPE) {
                  edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                                << " Unknown CommissioningTask"
                                                << " Unable to create FedCablingTask object!";
                } else {
                  edm::LogWarning(mlDqmSource_)
                      << "[SiStripCommissioningSource::" << __func__ << "]"
                      << " Unexpected CommissioningTask: " << SiStripEnumsAndStrings::runType(task_)
                      << " Unable to create FedCablingTask object!";
                }

                // Check if key is found and, if so, book histos and set update freq
                if (cablingTasks_.find(key) != cablingTasks_.end()) {
                  if (cablingTasks_[key]) {
                    cablingTasks_[key]->bookHistograms();
                    cablingTasks_[key]->updateFreq(1);  //@@ hardwired to update every event!!!
                    booked++;
                    std::stringstream ss;
                    ss << "[SiStripCommissioningSource::" << __func__ << "]"
                       << " Booking histograms for '" << cablingTasks_[key]->myName() << "' object with key 0x"
                       << std::hex << std::setfill('0') << std::setw(8) << key << std::dec << " in directory \"" << dir
                       << "\"";
                    LogTrace(mlDqmSource_) << ss.str();
                  } else {
                    std::stringstream ss;
                    ss << "[SiStripCommissioningSource::" << __func__ << "]"
                       << " NULL pointer to CommissioningTask for key 0x" << std::hex << std::setfill('0')
                       << std::setw(8) << key << std::dec << " in directory " << dir << " Unable to book histograms!";
                    edm::LogWarning(mlDqmSource_) << ss.str();
                  }
                } else {
                  std::stringstream ss;
                  ss << "[SiStripCommissioningSource::" << __func__ << "]"
                     << " Unable to find CommissioningTask for key 0x" << std::hex << std::setfill('0') << std::setw(8)
                     << key << std::dec << " in directory " << dir << " Unable to book histograms!";
                  edm::LogWarning(mlDqmSource_) << ss.str();
                }

              } else {
                std::stringstream ss;
                ss << "[SiStripCommissioningSource::" << __func__ << "]"
                   << " CommissioningTask object already exists for key 0x" << std::hex << std::setfill('0')
                   << std::setw(8) << key << std::dec << " in directory " << dir
                   << " Unable to create FedCablingTask object!";
                edm::LogWarning(mlDqmSource_) << ss.str();
              }

            }  // loop through apv pairs
          }    // loop through modules
        }      // loop through ccus
      }        // loop through rings
    }          // loop through fecs
  }            // loop through crates

  edm::LogVerbatim(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                 << " Created " << booked << " CommissioningTask objects and booked histograms";
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createTasks(sistrip::RunType run_type, const edm::EventSetup& setup) {
  uint16_t booked = 0;

  // latency has partition-level histos, so treat separately
  if (task_ == sistrip::APV_LATENCY) {
    for (uint16_t partition = 0; partition < 4; ++partition) {
      // make a task for every partition; tracker-wide histo is shared
      tasks_[0][partition] = new LatencyTask(
          dqm(),
          FedChannelConnection(
              partition + 1, sistrip::invalid_, sistrip::invalid_, sistrip::invalid_, sistrip::invalid_));
      tasks_[0][partition]->eventSetup(&setup);
      tasks_[0][partition]->bookHistograms();
      tasks_[0][partition]->updateFreq(updateFreq_);
      booked++;
    }

    // fine-delay has 1 histo for the whole tracker, so treat separately
  } else if (task_ == sistrip::FINE_DELAY) {
    tasks_[0][0] = new FineDelayTask(dqm(), FedChannelConnection());
    tasks_[0][0]->eventSetup(&setup);
    tasks_[0][0]->bookHistograms();
    tasks_[0][0]->updateFreq(updateFreq_);
    booked++;

  } else {  // now do any other task

    // Iterate through FED ids and channels
    for (auto ifed = fedCabling_->fedIds().begin(); ifed != fedCabling_->fedIds().end(); ++ifed) {
      // Iterate through connected FED channels
      auto conns = fedCabling_->fedConnections(*ifed);
      for (auto iconn = conns.begin(); iconn != conns.end(); ++iconn) {
        // Create FEC key
        SiStripFecKey fec_key(
            iconn->fecCrate(), iconn->fecSlot(), iconn->fecRing(), iconn->ccuAddr(), iconn->ccuChan());
        // Create FED key and check if non-zero
        SiStripFedKey fed_key(
            iconn->fedId(), SiStripFedKey::feUnit(iconn->fedCh()), SiStripFedKey::feChan(iconn->fedCh()));
        if (!iconn->isConnected()) {
          continue;
        }

        // define the view in which to work and paths for histograms
        //   currently FecView (control view) and FedView (readout view)
        //   DetView (detector view) implementation has started
        // Set working directory prior to booking histograms
        std::stringstream dir;
        dir << base_;
        if (view_ == "Default") {  // default
          if (run_type == sistrip::FAST_CABLING) {
            dir << fed_key.path();  // cabling in fed view
          } else {
            dir << fec_key.path();  // all other runs in control view
          }
        } else if (view_ == "FecView") {
          dir << fec_key.path();
        } else if (view_ == "FedView") {
          dir << fed_key.path();
        } else if (view_ == "DetView") {
          // currently just by detid from the connection, which is empty...
          dir << sistrip::root_ << sistrip::dir_ << sistrip::detectorView_ << sistrip::dir_ << iconn->detId();
        } else {
          edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                        << " Invalid view " << view_ << std::endl
                                        << " Histograms will end up all in the top directory.";
        }  // end if view_ == ...
        dqm()->setCurrentFolder(dir.str());

        // Create commissioning task objects
        if (!tasks_[iconn->fedId()][iconn->fedCh()]) {
          if (task_ == sistrip::FAST_CABLING) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new FastFedCablingTask(dqm(), *iconn);
          } else if (task_ == sistrip::APV_TIMING) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new ApvTimingTask(dqm(), *iconn);
          } else if (task_ == sistrip::FED_TIMING) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new FedTimingTask(dqm(), *iconn);
          } else if (task_ == sistrip::OPTO_SCAN) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new OptoScanTask(dqm(), *iconn);
          } else if (task_ == sistrip::VPSP_SCAN) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new VpspScanTask(dqm(), *iconn);
          } else if (task_ == sistrip::PEDESTALS) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new PedestalsTask(dqm(), *iconn);
          } else if (task_ == sistrip::PEDS_ONLY) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new PedsOnlyTask(dqm(), *iconn);
          } else if (task_ == sistrip::NOISE) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new NoiseTask(dqm(), *iconn, pedestalToken_, noiseToken_);
          } else if (task_ == sistrip::PEDS_FULL_NOISE) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new PedsFullNoiseTask(dqm(), *iconn, parameters_);
          } else if (task_ == sistrip::DAQ_SCOPE_MODE) {
            tasks_[iconn->fedId()][iconn->fedCh()] = new DaqScopeModeTask(dqm(), *iconn, parameters_);
          } else if (task_ == sistrip::CALIBRATION_SCAN || task_ == sistrip::CALIBRATION_SCAN_DECO) {
            tasks_[iconn->fedId()][iconn->fedCh()] =
                new CalibrationScanTask(dqm(), *iconn, task_, filename_.c_str(), run_, setup.getData(pedestalToken_));
          } else if (task_ == sistrip::CALIBRATION || task_ == sistrip::CALIBRATION_DECO) {
            tasks_[iconn->fedId()][iconn->fedCh()] =
                new CalibrationTask(dqm(), *iconn, task_, filename_.c_str(), run_, setup.getData(pedestalToken_));
          } else if (task_ == sistrip::UNDEFINED_RUN_TYPE) {
            edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                          << " Undefined CommissioningTask"
                                          << " Unable to create CommissioningTask object!";
          } else {
            edm::LogWarning(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                          << " Unknown CommissioningTask"
                                          << " Unable to create CommissioningTask object!";
          }

          // Check if fed_key is found and, if so, book histos and set update freq
          if (tasks_[iconn->fedId()][iconn->fedCh()]) {
            tasks_[iconn->fedId()][iconn->fedCh()]->eventSetup(&setup);

            if (task_ != sistrip::CALIBRATION_SCAN and task_ != sistrip::CALIBRATION_SCAN_DECO and
                task_ != sistrip::CALIBRATION and task_ != sistrip::CALIBRATION_DECO)
              tasks_[iconn->fedId()][iconn->fedCh()]->bookHistograms();
            else {
              if (task_ == sistrip::CALIBRATION_SCAN or task_ == sistrip::CALIBRATION_SCAN_DECO)
                static_cast<CalibrationScanTask*>(tasks_[iconn->fedId()][iconn->fedCh()])->setCurrentFolder(dir.str());
              else if (task_ == sistrip::CALIBRATION or task_ == sistrip::CALIBRATION_DECO)
                static_cast<CalibrationTask*>(tasks_[iconn->fedId()][iconn->fedCh()])->setCurrentFolder(dir.str());
            }
            tasks_[iconn->fedId()][iconn->fedCh()]->updateFreq(updateFreq_);
            booked++;
          } else {
            std::stringstream ss;
            ss << "[SiStripCommissioningSource::" << __func__ << "]"
               << " NULL pointer to CommissioningTask for key 0x" << std::hex << std::setfill('0') << std::setw(8)
               << fed_key.key() << std::dec << " in directory " << dir.str() << " Unable to book histograms!";
            edm::LogWarning(mlDqmSource_) << ss.str();
          }
        } else {
          std::stringstream ss;
          ss << "[SiStripCommissioningSource::" << __func__ << "]"
             << " CommissioningTask object already exists for key 0x" << std::hex << std::setfill('0') << std::setw(8)
             << fed_key.key() << std::dec << " in directory " << dir.str()
             << " Unable to create CommissioningTask object!";
          edm::LogWarning(mlDqmSource_) << ss.str();
        }
      }  // loop over fed channels
    }    // loop over feds
  }      // end other tasks
  edm::LogVerbatim(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]"
                                 << " Created " << booked << " CommissioningTask objects and booked histograms";
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::clearCablingTasks() {
  if (cablingTasks_.empty()) {
    return;
  }
  for (TaskMap::iterator itask = cablingTasks_.begin(); itask != cablingTasks_.end(); itask++) {
    if (itask->second) {
      delete itask->second;
    }
  }
  cablingTasks_.clear();
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::clearTasks() {
  if (tasks_.empty()) {
    return;
  }
  VecOfVecOfTasks::iterator ifed = tasks_.begin();
  for (; ifed != tasks_.end(); ifed++) {
    VecOfTasks::iterator ichan = ifed->begin();
    for (; ichan != ifed->end(); ichan++) {
      if (*ichan) {
        delete *ichan;
        *ichan = 0;
      }
    }
    ifed->resize(96, nullptr);
  }
  tasks_.resize(1024);
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::remove() {
  // TODO: remove no longer supported in DQMStore.
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::directory(std::stringstream& dir, uint32_t run_number) {
  // Get details about host
  char hn[256];
  gethostname(hn, sizeof(hn));
  struct hostent* he;
  he = gethostbyname(hn);

  // Extract host name and ip
  std::string host_name;
  std::string host_ip;
  if (he) {
    host_name = std::string(he->h_name);
    host_ip = std::string(inet_ntoa(*(struct in_addr*)(he->h_addr)));
  } else {
    host_name = "unknown.cern.ch";
    host_ip = "255.255.255.255";
  }

  // Reformat IP address
  std::string::size_type pos = 0;
  std::stringstream ip;
  //for ( uint16_t ii = 0; ii < 4; ++ii ) {
  while (pos != std::string::npos) {
    std::string::size_type tmp = host_ip.find('.', pos);
    if (tmp != std::string::npos) {
      ip << std::setw(3) << std::setfill('0') << host_ip.substr(pos, tmp - pos) << ".";
      pos = tmp + 1;  // skip the delimiter "."
    } else {
      ip << std::setw(3) << std::setfill('0') << host_ip.substr(pos);
      pos = std::string::npos;
    }
  }

  // Get pid
  pid_t pid = getpid();

  // Construct string
  if (run_number) {
    dir << std::setw(8) << std::setfill('0') << run_number << "_";
  }
  dir << ip.str() << "_" << std::setw(5) << std::setfill('0') << pid;
}

// -----------------------------------------------------------------------------
//
// void SiStripCommissioningSource::cablingForConnectionRun( const sistrip::RunType& type ) {

//   if ( type == sistrip::FED_CABLING ||
//        type == sistrip::QUITE_FAST_CABLING ||
//        type == sistrip::FAST_CABLING ) {
//     std::stringstream ss;
//     ss << "[SiStripCommissioningSource::" << __func__ << "]"
//        << " Run type is " << SiStripEnumsAndStrings::runType( type ) << "!"
//        << " Checking if cabling should be rebuilt using FED and device descriptions!...";
//     edm::LogVerbatim(mlDqmSource_) << ss.str();
//   } else { return; }

//   // Build and retrieve SiStripConfigDb object using service
//   SiStripConfigDb* db = edm::Service<SiStripConfigDb>().operator->(); //@@ NOT GUARANTEED TO BE THREAD SAFE!
//   LogTrace(mlCabling_)
//     << "[SiStripCommissioningSource::" << __func__ << "]"
//     << " Nota bene: using the SiStripConfigDb API"
//     << " as a \"service\" does not presently guarantee"
//     << " thread-safe behaviour!...";

//   if ( !db ) {
//     edm::LogError(mlCabling_)
//       << "[SiStripCommissioningSource::" << __func__ << "]"
//       << " NULL pointer to SiStripConfigDb returned by service!"
//       << " Cannot check if cabling needs to be rebuilt!";
//     return;
//   }

//   if ( db->getFedConnections().empty() ) {
//     edm::LogVerbatim(mlCabling_)
//       << "[SiStripCommissioningSource::" << __func__ << "]"
//       << " Datbase does not contain FED connections!"
//       << " Do not need to rebuild cabling object based on FED and device descriptions!";
//    return;
//   }

//   if ( fecCabling_ ) { delete fecCabling_; fecCabling_ = 0; }
//   if ( fedCabling_ ) { delete fedCabling_; fedCabling_ = 0; }

//   // Build FEC cabling
//   fecCabling_ = new SiStripFecCabling();
//   SiStripConfigDb::DcuDetIdMap mapping;
//   SiStripFedCablingBuilderFromDb::buildFecCablingFromDevices( db,
// 							      *fecCabling_,
// 							      mapping );
//   // Build FED cabling
//   fedCabling_ = new SiStripFedCabling();
//   SiStripFedCablingBuilderFromDb::getFedCabling( *fecCabling_,
// 						 *fedCabling_ );

//   edm::LogVerbatim(mlCabling_)
//     << "[SiStripCommissioningSource::" << __func__ << "]"
//     << " Cabling object rebuilt using on FED and device descriptions!";

// }

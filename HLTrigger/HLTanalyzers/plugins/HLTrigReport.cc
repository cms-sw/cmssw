/** \class HLTrigReport
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include <iomanip>
#include <cstring>
#include <sstream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigReportService.h"
#include "Math/QuantFuncMathCore.h"

#include "HLTrigReport.h"

HLTrigReport::ReportEvery HLTrigReport::decode(const std::string& value) {
  if (value == "never")
    return NEVER;

  if (value == "job")
    return EVERY_JOB;

  if (value == "run")
    return EVERY_RUN;

  if (value == "lumi")
    return EVERY_LUMI;

  if (value == "event")
    return EVERY_EVENT;

  throw cms::Exception("Configuration") << "Invalid option value \"" << value
                                        << "\". Legal values are \"job\", \"run\", \"lumi\", \"event\" and \"never\".";
}

//
// constructors and destructor
//
hltrigreport::Accumulate::Accumulate()
    : nEvents_(0),
      nWasRun_(0),
      nAccept_(0),
      nErrors_(0),
      hlWasRun_(0),
      hltL1s_(0),
      hltPre_(0),
      hlAccept_(0),
      hlAccTot_(0),
      hlErrors_(0),
      hlAccTotDS_(0),
      dsAccTotS_(0) {}

hltrigreport::Accumulate::Accumulate(size_t numHLNames,
                                     std::vector<std::vector<unsigned int> > const& hlIndex,
                                     std::vector<std::vector<unsigned int> > const& dsIndex)
    : nEvents_(0),
      nWasRun_(0),
      nAccept_(0),
      nErrors_(0),
      hlWasRun_(numHLNames),
      hltL1s_(numHLNames),
      hltPre_(numHLNames),
      hlAccept_(numHLNames),
      hlAccTot_(numHLNames),
      hlErrors_(numHLNames),
      hlAccTotDS_(hlIndex.size()),
      hlAllTotDS_(hlIndex.size()),
      dsAccTotS_(dsIndex.size()),
      dsAllTotS_(dsIndex.size()) {
  for (size_t ds = 0; ds < hlIndex.size(); ++ds) {
    hlAccTotDS_[ds].resize(hlIndex[ds].size());
  }

  for (size_t s = 0; s < dsIndex.size(); ++s) {
    dsAccTotS_[s].resize(dsIndex[s].size());
  }
}

void hltrigreport::Accumulate::accumulate(hltrigreport::Accumulate const& iOther) {
  nEvents_ += iOther.nEvents_;
  nWasRun_ += iOther.nWasRun_;
  nAccept_ += iOther.nAccept_;
  nErrors_ += iOther.nErrors_;

  auto vsum = [](auto& to, auto const& from) {
    for (size_t i = 0; i < from.size(); ++i) {
      to[i] += from[i];
    }
  };

  assert(hlWasRun_.size() == iOther.hlWasRun_.size());
  vsum(hlWasRun_, iOther.hlWasRun_);
  vsum(hltL1s_, iOther.hltL1s_);
  vsum(hltPre_, iOther.hltPre_);
  vsum(hlAccept_, iOther.hlAccept_);
  vsum(hlAccTot_, iOther.hlAccTot_);
  vsum(hlErrors_, iOther.hlErrors_);

  assert(hlAllTotDS_.size() == iOther.hlAllTotDS_.size());
  vsum(hlAllTotDS_, iOther.hlAllTotDS_);
  vsum(dsAllTotS_, iOther.dsAllTotS_);

  auto vvsum = [](auto& to, auto const& from) {
    for (size_t i = 0; i < from.size(); ++i) {
      assert(from[i].size() == to[i].size());
      for (size_t j = 0; j < from[i].size(); ++j) {
        to[i][j] += from[i][j];
      }
    }
  };

  vvsum(hlAccTotDS_, iOther.hlAccTotDS_);
  vvsum(dsAccTotS_, iOther.dsAccTotS_);
}

void hltrigreport::Accumulate::reset() {
  nEvents_ = 0;
  nWasRun_ = 0;
  nAccept_ = 0;
  nErrors_ = 0;

  auto vreset = [](auto& to) { std::fill(to.begin(), to.end(), 0); };

  vreset(hlWasRun_);
  vreset(hltL1s_);
  vreset(hltPre_);
  vreset(hlAccept_);
  vreset(hlAccTot_);
  vreset(hlErrors_);

  vreset(hlAllTotDS_);
  vreset(dsAllTotS_);

  auto vvreset = [&vreset](auto& to) {
    for (auto& e : to) {
      vreset(e);
    }
  };

  vvreset(hlAccTotDS_);
  vvreset(dsAccTotS_);
}

HLTrigReport::HLTrigReport(const edm::ParameterSet& iConfig)
    : hlTriggerResults_(iConfig.getParameter<edm::InputTag>("HLTriggerResults")),
      hlTriggerResultsToken_(consumes<edm::TriggerResults>(hlTriggerResults_)),
      configured_(false),
      hlNames_(0),
      hlIndex_(0),
      posL1s_(0),
      posPre_(0),
      datasetNames_(0),
      datasetContents_(0),
      isCustomDatasets_(false),
      dsIndex_(0),
      streamNames_(0),
      streamContents_(0),
      isCustomStreams_(false),
      refPath_("HLTriggerFinalPath"),
      refIndex_(0),
      refRate_(iConfig.getUntrackedParameter<double>("ReferenceRate", 100.0)),
      reportBy_(decode(iConfig.getUntrackedParameter<std::string>("reportBy", "job"))),
      resetBy_(decode(iConfig.getUntrackedParameter<std::string>("resetBy", "never"))),
      serviceBy_(decode(iConfig.getUntrackedParameter<std::string>("serviceBy", "never"))),
      hltConfig_() {
  const edm::ParameterSet customDatasets(
      iConfig.getUntrackedParameter<edm::ParameterSet>("CustomDatasets", edm::ParameterSet()));
  isCustomDatasets_ = (customDatasets != edm::ParameterSet());
  if (isCustomDatasets_) {
    datasetNames_ = customDatasets.getParameterNamesForType<std::vector<std::string> >();
    for (std::vector<std::string>::const_iterator name = datasetNames_.begin(); name != datasetNames_.end(); name++) {
      datasetContents_.push_back(customDatasets.getParameter<std::vector<std::string> >(*name));
    }
  }

  const edm::ParameterSet customStreams(
      iConfig.getUntrackedParameter<edm::ParameterSet>("CustomStreams", edm::ParameterSet()));
  isCustomStreams_ = (customStreams != edm::ParameterSet());
  if (isCustomStreams_) {
    streamNames_ = customStreams.getParameterNamesForType<std::vector<std::string> >();
    for (std::vector<std::string>::const_iterator name = streamNames_.begin(); name != streamNames_.end(); name++) {
      streamContents_.push_back(customStreams.getParameter<std::vector<std::string> >(*name));
    }
  }

  refPath_ = iConfig.getUntrackedParameter<std::string>("ReferencePath", "HLTriggerFinalPath");
  refIndex_ = 0;

  LogDebug("HLTrigReport") << "HL TiggerResults: " + hlTriggerResults_.encode()
                           << " using reference path and rate: " + refPath_ + " " << refRate_;

  if (serviceBy_ != NEVER and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->registerModule(this);
  }
}

HLTrigReport::~HLTrigReport() = default;

void HLTrigReport::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HLTriggerResults", edm::InputTag("TriggerResults", "", "HLT"));
  desc.addUntracked<std::string>("reportBy", "job");
  desc.addUntracked<std::string>("resetBy", "never");
  desc.addUntracked<std::string>("serviceBy", "never");

  edm::ParameterSetDescription customDatasetsParameters;
  desc.addUntracked<edm::ParameterSetDescription>("CustomDatasets", customDatasetsParameters);
  edm::ParameterSetDescription customStreamsParameters;
  desc.addUntracked<edm::ParameterSetDescription>("CustomStreams", customStreamsParameters);
  desc.addUntracked<std::string>("ReferencePath", "HLTriggerFinalPath");
  desc.addUntracked<double>("ReferenceRate", 100.0);

  descriptions.add("hltTrigReport", desc);
}

//
// member functions
//

const std::vector<std::string>& HLTrigReport::datasetNames() const { return datasetNames_; }
const std::vector<std::string>& HLTrigReport::streamNames() const { return streamNames_; }

void HLTrigReport::updateConfigCache() {
  // update trigger names
  hlNames_ = hltConfig_.triggerNames();

  const unsigned int n = hlNames_.size();

  // find the positions of seeding and prescaler modules
  posL1s_.resize(n);
  posPre_.resize(n);
  for (unsigned int i = 0; i < n; ++i) {
    posL1s_[i] = -1;
    posPre_[i] = -1;
    const std::vector<std::string>& moduleLabels(hltConfig_.moduleLabels(i));
    for (unsigned int j = 0; j < moduleLabels.size(); ++j) {
      const std::string& label = hltConfig_.moduleType(moduleLabels[j]);
      if (label == "HLTLevel1GTSeed")
        posL1s_[i] = j;
      else if (label == "HLTPrescaler")
        posPre_[i] = j;
    }
  }

  // if not overridden, reload the datasets and streams
  if (not isCustomDatasets_) {
    datasetNames_ = hltConfig_.datasetNames();
    datasetContents_ = hltConfig_.datasetContents();
  }
  if (not isCustomStreams_) {
    streamNames_ = hltConfig_.streamNames();
    streamContents_ = hltConfig_.streamContents();
  }

  // fill the matrices of hlIndex_, hlAccTotDS_
  hlIndex_.clear();
  hlIndex_.resize(datasetNames_.size());
  for (unsigned int ds = 0; ds < datasetNames_.size(); ds++) {
    unsigned int size = datasetContents_[ds].size();
    hlIndex_[ds].reserve(size);
    for (unsigned int p = 0; p < size; ++p) {
      unsigned int i = hltConfig_.triggerIndex(datasetContents_[ds][p]);
      if (i < n) {
        hlIndex_[ds].push_back(i);
      }
    }
  }

  // fill the matrices of dsIndex_, dsAccTotS_
  dsIndex_.clear();
  dsIndex_.resize(streamNames_.size());
  for (unsigned int s = 0; s < streamNames_.size(); ++s) {
    unsigned int size = streamContents_[s].size();
    dsIndex_.reserve(size);
    for (unsigned int ds = 0; ds < size; ++ds) {
      unsigned int i = 0;
      for (; i < datasetNames_.size(); i++)
        if (datasetNames_[i] == streamContents_[s][ds])
          break;
      // report only datasets that have at least one path otherwise crash
      if (i < datasetNames_.size() and !hlIndex_[i].empty()) {
        dsIndex_[s].push_back(i);
      }
    }
  }

  // if needed, update the reference path
  refIndex_ = hltConfig_.triggerIndex(refPath_);
  if (refIndex_ >= n) {
    refIndex_ = 0;
    edm::LogWarning("HLTrigReport") << "Requested reference path '" + refPath_ + "' not in HLT menu. "
                                    << "Using HLTriggerFinalPath instead.";
    refPath_ = "HLTriggerFinalPath";
    refIndex_ = hltConfig_.triggerIndex(refPath_);
    if (refIndex_ >= n) {
      refIndex_ = 0;
      edm::LogWarning("HLTrigReport") << "Requested reference path '" + refPath_ + "' not in HLT menu. "
                                      << "Using first path in table (index=0) instead.";
    }
  }

  if (serviceBy_ != NEVER and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->setDatasetNames(datasetNames_);
    edm::Service<HLTrigReportService>()->setStreamNames(streamNames_);
  }
}

void HLTrigReport::reset() { accumulate_.reset(); }

void HLTrigReport::beginJob() {
  if (resetBy_ == EVERY_JOB)
    reset();
}

void HLTrigReport::endJob() {
  if (reportBy_ == EVERY_JOB)
    dumpReport(accumulate_, "Summary for Job");
  if (serviceBy_ == EVERY_JOB) {
    updateService(accumulate_);
  }
}

void HLTrigReport::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, hlTriggerResults_.process(), changed)) {
    configured_ = true;
    if (changed) {
      dumpReport(accumulate_, "Summary for this HLT table");
      updateConfigCache();
      accumulate_ = Accumulate(hlNames_.size(), hlIndex_, dsIndex_);
    }
  } else {
    dumpReport(accumulate_, "Summary for this HLT table");
    // cannot initialize the HLT menu - reset and clear all counters and tables
    configured_ = false;

    accumulate_ = hltrigreport::Accumulate();
  }

  if (resetBy_ == EVERY_RUN)
    reset();
}

void HLTrigReport::endRun(edm::Run const& run, edm::EventSetup const& setup) {
  if (reportBy_ == EVERY_RUN) {
    std::stringstream stream;
    stream << "Summary for Run " << run.run();
    dumpReport(accumulate_, stream.str());
  }
  if (serviceBy_ == EVERY_RUN) {
    updateService(accumulate_);
  }
}

std::shared_ptr<HLTrigReport::Accumulate> HLTrigReport::globalBeginLuminosityBlock(edm::LuminosityBlock const& lumi,
                                                                                   edm::EventSetup const& setup) const {
  if (useLumiCache()) {
    if (not configured_) {
      return std::make_shared<Accumulate>();
    }
    return std::make_shared<Accumulate>(hlNames_.size(), hlIndex_, dsIndex_);
  }
  return std::shared_ptr<Accumulate>();
}

void HLTrigReport::globalEndLuminosityBlock(edm::LuminosityBlock const& lumi, edm::EventSetup const& setup) {
  if (not useLumiCache()) {
    return;
  }

  if (resetBy_ == EVERY_LUMI and readAfterLumi()) {
    //we will be reporting the last processed lumi
    accumulate_ = std::move(*luminosityBlockCache(lumi.index()));
  } else if (resetBy_ == NEVER or resetBy_ == EVERY_RUN or resetBy_ == EVERY_JOB or readAfterLumi()) {
    //we need to add this lumi's info to the longer lived accumulation
    accumulate_.accumulate(*luminosityBlockCache(lumi.index()));
  }

  if (reportBy_ == EVERY_LUMI) {
    std::stringstream stream;
    stream << "Summary for Run " << lumi.run() << ", LumiSection " << lumi.luminosityBlock();
    if (resetBy_ == EVERY_LUMI or resetBy_ == EVERY_EVENT) {
      dumpReport(*luminosityBlockCache(lumi.index()), stream.str());
    } else {
      dumpReport(accumulate_, stream.str());
    }
  }

  if (serviceBy_ == EVERY_LUMI) {
    if (resetBy_ == EVERY_LUMI or resetBy_ == EVERY_EVENT) {
      updateService(*luminosityBlockCache(lumi.index()));
    } else {
      updateService(accumulate_);
    }
  }
}

// ------------ method called to produce the data  ------------
void HLTrigReport::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // accumulation of statistics event by event

  using namespace std;
  using namespace edm;

  auto& accumulate = chooseAccumulate(iEvent.getLuminosityBlock().index());
  if (resetBy_ == EVERY_EVENT) {
    //NOTE if we have reportBy == lumi/run/job then we will report the last
    // event in each lumi/run/job
    accumulate.reset();
  }

  accumulate.nEvents_++;

  // get hold of TriggerResults
  Handle<TriggerResults> HLTR;
  iEvent.getByToken(hlTriggerResultsToken_, HLTR);
  if (HLTR.isValid()) {
    if (HLTR->wasrun())
      accumulate.nWasRun_++;
    const bool accept(HLTR->accept());
    LogDebug("HLTrigReport") << "HLT TriggerResults decision: " << accept;
    if (accept)
      ++accumulate.nAccept_;
    if (HLTR->error())
      accumulate.nErrors_++;
  } else {
    LogDebug("HLTrigReport") << "HLT TriggerResults with label [" + hlTriggerResults_.encode() + "] not found!";
    accumulate.nErrors_++;
    return;
  }

  // HLTConfigProvider not configured - cannot produce any detailed statistics
  if (not configured_)
    return;

  // decision for each HL algorithm
  const unsigned int n(hlNames_.size());
  bool acceptedByPrevoiusPaths = false;
  for (unsigned int i = 0; i != n; ++i) {
    if (HLTR->wasrun(i))
      accumulate.hlWasRun_[i]++;
    if (HLTR->accept(i)) {
      acceptedByPrevoiusPaths = true;
      accumulate.hlAccept_[i]++;
    }
    if (acceptedByPrevoiusPaths)
      accumulate.hlAccTot_[i]++;
    if (HLTR->error(i))
      accumulate.hlErrors_[i]++;
    const int index(static_cast<int>(HLTR->index(i)));
    if (HLTR->accept(i)) {
      if (index >= posL1s_[i])
        accumulate.hltL1s_[i]++;
      if (index >= posPre_[i])
        accumulate.hltPre_[i]++;
    } else {
      if (index > posL1s_[i])
        accumulate.hltL1s_[i]++;
      if (index > posPre_[i])
        accumulate.hltPre_[i]++;
    }
  }

  // calculate accumulation of accepted events by a path within a dataset
  std::vector<bool> acceptedByDS(hlIndex_.size(), false);
  for (size_t ds = 0; ds < hlIndex_.size(); ++ds) {
    for (size_t p = 0; p < hlIndex_[ds].size(); ++p) {
      if (acceptedByDS[ds] or HLTR->accept(hlIndex_[ds][p])) {
        acceptedByDS[ds] = true;
        accumulate.hlAccTotDS_[ds][p]++;
      }
    }
    if (acceptedByDS[ds])
      accumulate.hlAllTotDS_[ds]++;
  }

  // calculate accumulation of accepted events by a dataset within a stream
  for (size_t s = 0; s < dsIndex_.size(); ++s) {
    bool acceptedByS = false;
    for (size_t ds = 0; ds < dsIndex_[s].size(); ++ds) {
      if (acceptedByS or acceptedByDS[dsIndex_[s][ds]]) {
        acceptedByS = true;
        accumulate.dsAccTotS_[s][ds]++;
      }
    }
    if (acceptedByS)
      accumulate.dsAllTotS_[s]++;
  }

  if (reportBy_ == EVERY_EVENT) {
    std::stringstream stream;
    stream << "Summary for Run " << iEvent.run() << ", LumiSection " << iEvent.luminosityBlock() << ", Event "
           << iEvent.id();
    dumpReport(accumulate, stream.str());
  }
  if (serviceBy_ == EVERY_EVENT) {
    updateService(accumulate);
  }
}

void HLTrigReport::updateService(Accumulate const& accumulate) const {
  edm::Service<HLTrigReportService> s;
  if (s) {
    s->setDatasetCounts(accumulate.hlAllTotDS_);
    s->setStreamCounts(accumulate.dsAllTotS_);
  }
}

void HLTrigReport::dumpReport(hltrigreport::Accumulate const& accumulate,
                              std::string const& header /* = std::string() */) const {
  // final printout of accumulated statistics

  using namespace std;
  using namespace edm;
  const unsigned int n(hlNames_.size());

  if ((n == 0) and (accumulate.nEvents_ == 0))
    return;

  LogVerbatim("HLTrigReport") << dec << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report "
                              << "---------- Event  Summary ------------" << endl;
  if (not header.empty())
    LogVerbatim("HLTrigReport") << "HLT-Report " << header << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report"
                              << " Events total = " << accumulate.nEvents_ << " wasrun = " << accumulate.nWasRun_
                              << " passed = " << accumulate.nAccept_ << " errors = " << accumulate.nErrors_ << endl;

  // HLTConfigProvider not configured - cannot produce any detailed statistics
  if (not configured_)
    return;

  double scale = accumulate.hlAccept_[refIndex_] > 0 ? refRate_ / accumulate.hlAccept_[refIndex_] : 0.;
  double alpha = 1 - (1.0 - .6854) / 2;  // for the Clopper-Pearson 68% CI

  LogVerbatim("HLTrigReport") << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report "
                              << "---------- HLTrig Summary ------------" << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report " << right << setw(7) << "HLT #"
                              << " " << right << setw(7) << "WasRun"
                              << " " << right << setw(7) << "L1S"
                              << " " << right << setw(7) << "Pre"
                              << " " << right << setw(7) << "HLT"
                              << " " << right << setw(9) << "%L1sPre"
                              << " " << right << setw(7) << "Rate"
                              << " " << right << setw(7) << "RateHi"
                              << " " << right << setw(7) << "Errors"
                              << " "
                              << "Name" << endl;

  if (n > 0) {
    for (unsigned int i = 0; i != n; ++i) {
      LogVerbatim("HLTrigReport")
          << "HLT-Report " << right << setw(7) << i << " " << right << setw(7) << accumulate.hlWasRun_[i] << " "
          << right << setw(7) << accumulate.hltL1s_[i] << " " << right << setw(7) << accumulate.hltPre_[i] << " "
          << right << setw(7) << accumulate.hlAccept_[i] << " " << right << setw(9) << fixed << setprecision(5)
          << static_cast<float>(100 * accumulate.hlAccept_[i]) / static_cast<float>(max(accumulate.hltPre_[i], 1u))
          << " " << right << setw(7) << fixed << setprecision(1) << scale * accumulate.hlAccept_[i] << " " << right
          << setw(7) << fixed << setprecision(1)
          << ((accumulate.hlAccept_[refIndex_] - accumulate.hlAccept_[i] > 0)
                  ? refRate_ * ROOT::Math::beta_quantile(alpha,
                                                         accumulate.hlAccept_[i] + 1,
                                                         accumulate.hlAccept_[refIndex_] - accumulate.hlAccept_[i])
                  : 0)
          << " " << right << setw(7) << accumulate.hlErrors_[i] << " " << hlNames_[i] << endl;
    }
  }

  LogVerbatim("HLTrigRprtTt") << endl;
  LogVerbatim("HLTrigRprtTt") << "HLT-Report "
                              << "---------- HLTrig Summary ------------" << endl;
  LogVerbatim("HLTrigRprtTt") << "HLT-Report " << right << setw(7) << "HLT #"
                              << " " << right << setw(7) << "WasRun"
                              << " " << right << setw(7) << "L1S"
                              << " " << right << setw(7) << "Pre"
                              << " " << right << setw(7) << "HLT"
                              << " " << right << setw(9) << "%L1sPre"
                              << " " << right << setw(7) << "Rate"
                              << " " << right << setw(7) << "RateHi"
                              << " " << right << setw(7) << "HLTtot"
                              << " " << right << setw(7) << "RateTot"
                              << " " << right << setw(7) << "Errors"
                              << " "
                              << "Name" << endl;

  if (n > 0) {
    for (unsigned int i = 0; i != n; ++i) {
      LogVerbatim("HLTrigRprtTt")
          << "HLT-Report " << right << setw(7) << i << " " << right << setw(7) << accumulate.hlWasRun_[i] << " "
          << right << setw(7) << accumulate.hltL1s_[i] << " " << right << setw(7) << accumulate.hltPre_[i] << " "
          << right << setw(7) << accumulate.hlAccept_[i] << " " << right << setw(9) << fixed << setprecision(5)
          << static_cast<float>(100 * accumulate.hlAccept_[i]) / static_cast<float>(max(accumulate.hltPre_[i], 1u))
          << " " << right << setw(7) << fixed << setprecision(1) << scale * accumulate.hlAccept_[i] << " " << right
          << setw(7) << fixed << setprecision(1)
          << ((accumulate.hlAccept_[refIndex_] - accumulate.hlAccept_[i] > 0)
                  ? refRate_ * ROOT::Math::beta_quantile(alpha,
                                                         accumulate.hlAccept_[i] + 1,
                                                         accumulate.hlAccept_[refIndex_] - accumulate.hlAccept_[i])
                  : 0)
          << " " << right << setw(7) << accumulate.hlAccTot_[i] << " " << right << setw(7) << fixed << setprecision(1)
          << scale * accumulate.hlAccTot_[i] << " " << right << setw(7) << accumulate.hlErrors_[i] << " " << hlNames_[i]
          << endl;
    }

    // now for each dataset
    for (size_t ds = 0; ds < hlIndex_.size(); ++ds) {
      LogVerbatim("HLTrigRprtPD") << endl;
      LogVerbatim("HLTrigRprtPD") << "HLT-Report "
                                  << "---------- Dataset Summary: " << datasetNames_[ds] << " ------------"
                                  << accumulate.hlAllTotDS_[ds] << endl;
      LogVerbatim("HLTrigRprtPD") << "HLT-Report " << right << setw(7) << "HLT #"
                                  << " " << right << setw(7) << "WasRun"
                                  << " " << right << setw(7) << "L1S"
                                  << " " << right << setw(7) << "Pre"
                                  << " " << right << setw(7) << "HLT"
                                  << " " << right << setw(9) << "%L1sPre"
                                  << " " << right << setw(7) << "Rate"
                                  << " " << right << setw(7) << "RateHi"
                                  << " " << right << setw(7) << "HLTtot"
                                  << " " << right << setw(7) << "RateTot"
                                  << " " << right << setw(7) << "Errors"
                                  << " "
                                  << "Name" << endl;
      for (size_t p = 0; p < hlIndex_[ds].size(); ++p) {
        LogVerbatim("HLTrigRprtPD")
            << "HLT-Report " << right << setw(7) << p << " " << right << setw(7)
            << accumulate.hlWasRun_[hlIndex_[ds][p]] << " " << right << setw(7) << accumulate.hltL1s_[hlIndex_[ds][p]]
            << " " << right << setw(7) << accumulate.hltPre_[hlIndex_[ds][p]] << " " << right << setw(7)
            << accumulate.hlAccept_[hlIndex_[ds][p]] << " " << right << setw(9) << fixed << setprecision(5)
            << static_cast<float>(100 * accumulate.hlAccept_[hlIndex_[ds][p]]) /
                   static_cast<float>(max(accumulate.hltPre_[hlIndex_[ds][p]], 1u))
            << " " << right << setw(7) << fixed << setprecision(1) << scale * accumulate.hlAccept_[hlIndex_[ds][p]]
            << " " << right << setw(7) << fixed << setprecision(1)
            << ((accumulate.hlAccept_[refIndex_] - accumulate.hlAccept_[hlIndex_[ds][p]] > 0)
                    ? refRate_ * ROOT::Math::beta_quantile(
                                     alpha,
                                     accumulate.hlAccept_[hlIndex_[ds][p]] + 1,
                                     accumulate.hlAccept_[refIndex_] - accumulate.hlAccept_[hlIndex_[ds][p]])
                    : 0)
            << " " << right << setw(7) << accumulate.hlAccTotDS_[ds][p] << " " << right << setw(7) << fixed
            << setprecision(1) << scale * accumulate.hlAccTotDS_[ds][p] << " " << right << setw(7)
            << accumulate.hlErrors_[hlIndex_[ds][p]] << " " << hlNames_[hlIndex_[ds][p]] << endl;
      }
    }

    // now for each stream
    for (size_t s = 0; s < dsIndex_.size(); ++s) {
      LogVerbatim("HLTrigRprtST") << endl;
      LogVerbatim("HLTrigRprtST") << "HLT-Report "
                                  << "---------- Stream Summary: " << streamNames_[s] << " ------------"
                                  << accumulate.dsAllTotS_[s] << endl;
      LogVerbatim("HLTrigRprtST") << "HLT-Report " << right << setw(10) << "Dataset #"
                                  << " " << right << setw(10) << "Individual"
                                  << " " << right << setw(10) << "Total"
                                  << " " << right << setw(10) << "Rate"
                                  << " " << right << setw(10) << "RateHi"
                                  << " " << right << setw(10) << "RateTot"
                                  << " "
                                  << "Name" << endl;
      for (size_t ds = 0; ds < dsIndex_[s].size(); ++ds) {
        unsigned int acceptedDS = accumulate.hlAccTotDS_[dsIndex_[s][ds]][hlIndex_[dsIndex_[s][ds]].size() - 1];
        LogVerbatim("HLTrigRprtST")
            << "HLT-Report " << right << setw(10) << ds << " " << right << setw(10) << acceptedDS << " " << right
            << setw(10) << accumulate.dsAccTotS_[s][ds] << " " << right << setw(10) << fixed << setprecision(1)
            << scale * acceptedDS << " " << right << setw(10) << fixed << setprecision(1)
            << ((accumulate.hlAccept_[refIndex_] - acceptedDS > 0)
                    ? refRate_ *
                          ROOT::Math::beta_quantile(alpha, acceptedDS + 1, accumulate.hlAccept_[refIndex_] - acceptedDS)
                    : 0)
            << " " << right << setw(10) << fixed << setprecision(1) << scale * accumulate.dsAccTotS_[s][ds] << " "
            << datasetNames_[dsIndex_[s][ds]] << endl;
      }
    }

  } else {
    LogVerbatim("HLTrigReport") << "HLT-Report - No HLT paths found!" << endl;
  }

  LogVerbatim("HLTrigReport") << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report end!" << endl;
  LogVerbatim("HLTrigReport") << endl;

  return;
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTrigReport);

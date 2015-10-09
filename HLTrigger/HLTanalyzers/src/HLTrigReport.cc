/** \class HLTrigReport
 *
 * See header file for documentation
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTanalyzers/interface/HLTrigReport.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Math/QuantFuncMathCore.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTanalyzers/interface/HLTrigReportService.h"

#include <iomanip>
#include <cstring>
#include <sstream>


HLTrigReport::ReportEvery HLTrigReport::decode(const std::string & value) {
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
  
  throw cms::Exception("Configuration") << "Invalid option value \"" << value << "\". Legal values are \"job\", \"run\", \"lumi\", \"event\" and \"never\".";
}


//
// constructors and destructor
//
HLTrigReport::HLTrigReport(const edm::ParameterSet& iConfig) :
  hlTriggerResults_(iConfig.getParameter<edm::InputTag> ("HLTriggerResults")),
  hlTriggerResultsToken_(consumes<edm::TriggerResults>(hlTriggerResults_)),
  configured_(false),
  nEvents_(0),
  nWasRun_(0),
  nAccept_(0),
  nErrors_(0),
  hlWasRun_(0),
  hltL1s_(0),
  hltPre_(0),
  hlAccept_(0),
  hlAccTot_(0),
  hlErrors_(0),
  posL1s_(0),
  posPre_(0),
  hlNames_(0),
  hlIndex_(0),
  hlAccTotDS_(0),
  datasetNames_(0),
  datasetContents_(0),
  isCustomDatasets_(false),
  dsIndex_(0),
  dsAccTotS_(0),
  streamNames_(0),
  streamContents_(0),
  isCustomStreams_(false),
  refPath_("HLTriggerFinalPath"),
  refIndex_(0),
  refRate_(100.0),
  reportBy_( decode(iConfig.getUntrackedParameter<std::string>("reportBy",  "job")) ),
  resetBy_(  decode(iConfig.getUntrackedParameter<std::string>("resetBy",   "never")) ),
  serviceBy_(decode(iConfig.getUntrackedParameter<std::string>("serviceBy", "never")) ),
  hltConfig_()
{
  const edm::ParameterSet customDatasets(iConfig.getUntrackedParameter<edm::ParameterSet>("CustomDatasets", edm::ParameterSet()));
  isCustomDatasets_ = (customDatasets != edm::ParameterSet());
  if (isCustomDatasets_) {
    datasetNames_ = customDatasets.getParameterNamesForType<std::vector<std::string> >();
    for (std::vector<std::string>::const_iterator name = datasetNames_.begin(); name != datasetNames_.end(); name++) {
      datasetContents_.push_back(customDatasets.getParameter<std::vector<std::string> >(*name));
    }
  }

  const edm::ParameterSet customStreams (iConfig.getUntrackedParameter<edm::ParameterSet>("CustomStreams" , edm::ParameterSet()));
  isCustomStreams_  = (customStreams  != edm::ParameterSet());
  if (isCustomStreams_ ) {
    streamNames_ = customStreams.getParameterNamesForType<std::vector<std::string> >();
    for (std::vector<std::string>::const_iterator name = streamNames_.begin(); name != streamNames_.end(); name++) {
      streamContents_.push_back(customStreams.getParameter<std::vector<std::string> >(*name));
    }
  }

  refPath_ = iConfig.getUntrackedParameter<std::string>("ReferencePath","HLTriggerFinalPath");
  refRate_ = iConfig.getUntrackedParameter<double>("ReferenceRate", 100.0);
  refIndex_= 0;

  LogDebug("HLTrigReport")
    << "HL TiggerResults: " + hlTriggerResults_.encode()
    << " using reference path and rate: " + refPath_ + " " << refRate_;

  if (serviceBy_ != NEVER and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->registerModule(this);
  }

}

HLTrigReport::~HLTrigReport() { }

void
HLTrigReport::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HLTriggerResults",edm::InputTag("TriggerResults","","HLT"));
  desc.addUntracked<std::string>("reportBy","job");
  desc.addUntracked<std::string>("resetBy","never");
  desc.addUntracked<std::string>("serviceBy","never");

  edm::ParameterSetDescription customDatasetsParameters;
  desc.addUntracked<edm::ParameterSetDescription>("CustomDatasets" ,customDatasetsParameters);
  edm::ParameterSetDescription customStreamsParameters;
  desc.addUntracked<edm::ParameterSetDescription>("CustomStreams" ,customStreamsParameters);
  desc.addUntracked<std::string>("ReferencePath","HLTriggerFinalPath");
  desc.addUntracked<double>("ReferenceRate",100.0);

  descriptions.add("hltTrigReport",desc);
}

//
// member functions
//

const std::vector<std::string>& HLTrigReport::datasetNames() const {
  return datasetNames_;
}
const std::vector<std::string>& HLTrigReport::streamNames() const {
  return streamNames_;
}
const std::vector<unsigned int>& HLTrigReport::datasetCounts() const {
  return hlAllTotDS_;
}
const std::vector<unsigned int>& HLTrigReport::streamCounts() const {
  return dsAllTotS_;
}

void HLTrigReport::reset(bool changed /* = false */) {

  // reset global counters
  nEvents_ = 0;
  nWasRun_ = 0;
  nAccept_ = 0;
  nErrors_ = 0;

  // update trigger names
  if (changed)
    hlNames_ = hltConfig_.triggerNames();

  const unsigned int n = hlNames_.size();

  if (changed) {
    // resize per-path counters
    hlWasRun_.resize(n);
    hltL1s_.resize(n);
    hltPre_.resize(n);
    hlAccept_.resize(n);
    hlAccTot_.resize(n);
    hlErrors_.resize(n);
    // find the positions of seeding and prescaler modules
    posL1s_.resize(n);
    posPre_.resize(n);
    for (unsigned int i = 0; i < n; ++i) {
      posL1s_[i] = -1;
      posPre_[i] = -1;
      const std::vector<std::string> & moduleLabels(hltConfig_.moduleLabels(i));
      for (unsigned int j = 0; j < moduleLabels.size(); ++j) {
        const std::string & label = hltConfig_.moduleType(moduleLabels[j]);
        if (label == "HLTLevel1GTSeed")
          posL1s_[i] = j;
        else if (label == "HLTPrescaler")
          posPre_[i] = j;
      }
    }
  }

  // reset per-path counters
  for (unsigned int i = 0; i < n; ++i) {
    hlWasRun_[i] = 0;
    hltL1s_[i]   = 0;
    hltPre_[i]   = 0;
    hlAccept_[i] = 0;
    hlAccTot_[i] = 0;
    hlErrors_[i] = 0;
  }

  // if not overridden, reload the datasets and streams
  if (changed and not isCustomDatasets_) {
    datasetNames_    = hltConfig_.datasetNames();
    datasetContents_ = hltConfig_.datasetContents();
  }
  if (changed and not isCustomStreams_) {
    streamNames_     = hltConfig_.streamNames();
    streamContents_  = hltConfig_.streamContents();
  }

  if (changed) {
    // fill the matrices of hlIndex_, hlAccTotDS_
    hlIndex_.clear();
    hlIndex_.resize(datasetNames_.size());
    hlAccTotDS_.clear();
    hlAllTotDS_.clear();
    hlAccTotDS_.resize(datasetNames_.size());
    hlAllTotDS_.resize(datasetNames_.size());
    for (unsigned int ds = 0; ds < datasetNames_.size(); ds++) {
      unsigned int size = datasetContents_[ds].size();
      hlIndex_[ds].reserve(size);
      hlAccTotDS_[ds].reserve(size);
      hlAllTotDS_[ds]=0;
      for (unsigned int p = 0; p < size; ++p) {
        unsigned int i = hltConfig_.triggerIndex(datasetContents_[ds][p]);
        if (i<n) {
          hlIndex_[ds].push_back(i);
          hlAccTotDS_[ds].push_back(0);
        }
      }
    }
  } else {
    // reset the matrix of hlAccTotDS_
    for (unsigned int ds = 0; ds < datasetNames_.size(); ds++) {
      hlAllTotDS_[ds]=0;
      for (unsigned int i = 0; i < hlAccTotDS_[ds].size(); ++i)
          hlAccTotDS_[ds][i] = 0;
    }
  }

  if (changed) {
    // fill the matrices of dsIndex_, dsAccTotS_
    dsIndex_.clear();
    dsIndex_.resize(streamNames_.size());
    dsAccTotS_.clear();
    dsAllTotS_.clear();
    dsAccTotS_.resize(streamNames_.size());
    dsAllTotS_.resize(streamNames_.size());
    for (unsigned int s = 0; s < streamNames_.size(); ++s) {
      unsigned int size = streamContents_[s].size();
      dsIndex_.reserve(size);
      dsAccTotS_.reserve(size);
      dsAllTotS_[s]=0;
      for (unsigned int ds = 0; ds < size; ++ds) {
        unsigned int i = 0;
        for (; i<datasetNames_.size(); i++) if (datasetNames_[i] == streamContents_[s][ds]) 
          break;
        // report only datasets that have at least one path otherwise crash
        if (i < datasetNames_.size() and hlIndex_[i].size() > 0) {
          dsIndex_[s].push_back(i);
          dsAccTotS_[s].push_back(0);
        }
      }
    }
  } else {
    // reset the matrix of dsAccTotS_
    for (unsigned int s = 0; s < streamNames_.size(); ++s) {
      dsAllTotS_[s]=0;
      for (unsigned int i = 0; i < dsAccTotS_[s].size(); ++i)
        dsAccTotS_[s][i] = 0;
    }
  }

  // if needed, update the reference path
  if (changed) {
    refIndex_ = hltConfig_.triggerIndex(refPath_);
    if (refIndex_ >= n) {
      refIndex_ = 0;
      edm::LogWarning("HLTrigReport")
        << "Requested reference path '"+refPath_+"' not in HLT menu. "
        << "Using HLTriggerFinalPath instead.";
      refPath_ = "HLTriggerFinalPath";
      refIndex_ = hltConfig_.triggerIndex(refPath_);
      if (refIndex_ >= n) {
        refIndex_ = 0;
        edm::LogWarning("HLTrigReport")
          << "Requested reference path '"+refPath_+"' not in HLT menu. "
          << "Using first path in table (index=0) instead.";
      }
    }
  }

  if (changed and serviceBy_ != NEVER and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->setDatasetNames(datasetNames_);
    edm::Service<HLTrigReportService>()->setStreamNames(streamNames_);
  }

}

void HLTrigReport::beginJob() {
  if (resetBy_ == EVERY_JOB)
    reset();
}

void HLTrigReport::endJob() {
  if (reportBy_ == EVERY_JOB)
    dumpReport("Summary for Job");
  if (serviceBy_ == EVERY_JOB and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->setDatasetCounts(datasetCounts());
    edm::Service<HLTrigReportService>()->setStreamCounts(streamCounts());
  }

}

void
HLTrigReport::beginRun(edm::Run const & iRun, edm::EventSetup const& iSetup)
{
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, hlTriggerResults_.process(), changed)) {
    configured_ = true;
    if (changed) {
      dumpReport("Summary for this HLT table");
      reset(true);
    }
  } else {
    dumpReport("Summary for this HLT table");
    // cannot initialize the HLT menu - reset and clear all counters and tables
    configured_ = false;
    nEvents_    = 0;
    nWasRun_    = 0;
    nAccept_    = 0;
    nErrors_    = 0;
    hlWasRun_.clear();
    hltL1s_.clear();
    hltPre_.clear();
    hlAccept_.clear();
    hlAccTot_.clear();
    hlErrors_.clear();
    posL1s_.clear();
    posPre_.clear();
    hlNames_.clear();
    hlIndex_.clear();
    hlAccTotDS_.clear();
    hlAllTotDS_.clear();
    dsIndex_.clear();
    dsAccTotS_.clear();
    dsAllTotS_.clear();
  }

  if (resetBy_ == EVERY_RUN) reset();

}

void HLTrigReport::endRun(edm::Run const & run, edm::EventSetup const & setup) {
  if (reportBy_ == EVERY_RUN) {
    std::stringstream stream;
    stream << "Summary for Run " << run.run();
    dumpReport(stream.str());
  }
  if (serviceBy_ == EVERY_RUN and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->setDatasetCounts(datasetCounts());
    edm::Service<HLTrigReportService>()->setStreamCounts(streamCounts());
  }
}

void HLTrigReport::beginLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup) {
  if (resetBy_ == EVERY_LUMI) reset();
}

void HLTrigReport::endLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup) {
  if (reportBy_ == EVERY_LUMI) {
    std::stringstream stream;
    stream << "Summary for Run " << lumi.run() << ", LumiSection " << lumi.luminosityBlock();
    dumpReport(stream.str());
  }
  if (serviceBy_ == EVERY_LUMI and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->setDatasetCounts(datasetCounts());
    edm::Service<HLTrigReportService>()->setStreamCounts(streamCounts());
  }
}


// ------------ method called to produce the data  ------------
void
HLTrigReport::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // accumulation of statistics event by event

  using namespace std;
  using namespace edm;

  if (resetBy_ == EVERY_EVENT) reset();

  nEvents_++;

  // get hold of TriggerResults
  Handle<TriggerResults> HLTR;
  iEvent.getByToken(hlTriggerResultsToken_, HLTR);
  if (HLTR.isValid()) {
    if (HLTR->wasrun()) nWasRun_++;
    const bool accept(HLTR->accept());
    LogDebug("HLTrigReport") << "HLT TriggerResults decision: " << accept;
    if (accept) ++nAccept_;
    if (HLTR->error()) nErrors_++;
  } else {
    LogDebug("HLTrigReport") << "HLT TriggerResults with label ["+hlTriggerResults_.encode()+"] not found!";
    nErrors_++;
    return;
  }

  // HLTConfigProvider not configured - cannot produce any detailed statistics
  if (not configured_)
    return;

  // decision for each HL algorithm
  const unsigned int n(hlNames_.size());
  bool acceptedByPrevoiusPaths = false;
  for (unsigned int i=0; i!=n; ++i) {
    if (HLTR->wasrun(i)) hlWasRun_[i]++;
    if (HLTR->accept(i)) {
      acceptedByPrevoiusPaths = true;
      hlAccept_[i]++;
    }
    if (acceptedByPrevoiusPaths) hlAccTot_[i]++;
    if (HLTR->error(i) ) hlErrors_[i]++;
    const int index(static_cast<int>(HLTR->index(i)));
    if (HLTR->accept(i)) {
      if (index >= posL1s_[i]) hltL1s_[i]++;
      if (index >= posPre_[i]) hltPre_[i]++;
    } else {
      if (index >  posL1s_[i]) hltL1s_[i]++;
      if (index >  posPre_[i]) hltPre_[i]++;
    }
  }

  // calculate accumulation of accepted events by a path within a dataset
  std::vector<bool> acceptedByDS(hlIndex_.size(), false);
  for (size_t ds=0; ds<hlIndex_.size(); ++ds) {
    for (size_t p=0; p<hlIndex_[ds].size(); ++p) {
      if (acceptedByDS[ds] or HLTR->accept(hlIndex_[ds][p])) {
        acceptedByDS[ds] = true;
        hlAccTotDS_[ds][p]++;
      }
    }
    if (acceptedByDS[ds]) hlAllTotDS_[ds]++;
  }

  // calculate accumulation of accepted events by a dataset within a stream
  for (size_t s=0; s<dsIndex_.size(); ++s) {
    bool acceptedByS = false;
    for (size_t ds=0; ds<dsIndex_[s].size(); ++ds) {
      if (acceptedByS or acceptedByDS[dsIndex_[s][ds]]) {
        acceptedByS = true;
        dsAccTotS_[s][ds]++;
      }
    }
    if (acceptedByS) dsAllTotS_[s]++;
  }

  if (reportBy_ == EVERY_EVENT) {
    std::stringstream stream;
    stream << "Summary for Run " << iEvent.run() << ", LumiSection " << iEvent.luminosityBlock() << ", Event " << iEvent.id();
    dumpReport(stream.str());
  }
  if (serviceBy_ == EVERY_EVENT and edm::Service<HLTrigReportService>()) {
    edm::Service<HLTrigReportService>()->setDatasetCounts(datasetCounts());
    edm::Service<HLTrigReportService>()->setStreamCounts(streamCounts());
  }

}

void
HLTrigReport::dumpReport(std::string const & header /* = std::string() */)
{
  // final printout of accumulated statistics

  using namespace std;
  using namespace edm;
  const unsigned int n(hlNames_.size());

  if ((n==0) and (nEvents_==0)) return;

  LogVerbatim("HLTrigReport") << dec << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report " << "---------- Event  Summary ------------" << endl;
  if (not header.empty())
    LogVerbatim("HLTrigReport") << "HLT-Report " << header << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report"
	 << " Events total = " << nEvents_
	 << " wasrun = " << nWasRun_
	 << " passed = " << nAccept_
	 << " errors = " << nErrors_
	 << endl;

  // HLTConfigProvider not configured - cannot produce any detailed statistics
  if (not configured_)
    return;

  double scale = hlAccept_[refIndex_]>0 ? refRate_/hlAccept_[refIndex_] : 0.;
  double alpha = 1 - (1.0 - .6854)/2; // for the Clopper-Pearson 68% CI

  LogVerbatim("HLTrigReport") << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report " << "---------- HLTrig Summary ------------" << endl;
  LogVerbatim("HLTrigReport") << "HLT-Report "
	 << right << setw(7) << "HLT #" << " "
	 << right << setw(7) << "WasRun" << " "
	 << right << setw(7) << "L1S" << " "
	 << right << setw(7) << "Pre" << " "
	 << right << setw(7) << "HLT" << " "
	 << right << setw(9) << "%L1sPre" << " "
         << right << setw(7) << "Rate" << " "
         << right << setw(7) << "RateHi" << " "
	 << right << setw(7) << "Errors" << " "
	 << "Name" << endl;

  if (n>0) {
    for (unsigned int i=0; i!=n; ++i) {
      LogVerbatim("HLTrigReport") << "HLT-Report "
	   << right << setw(7) << i << " "
	   << right << setw(7) << hlWasRun_[i] << " "
	   << right << setw(7) << hltL1s_[i] << " "
	   << right << setw(7) << hltPre_[i] << " "
	   << right << setw(7) << hlAccept_[i] << " "
	   << right << setw(9) << fixed << setprecision(5)
	   << static_cast<float>(100*hlAccept_[i])/
	      static_cast<float>(max(hltPre_[i], 1u)) << " "
           << right << setw(7) << fixed << setprecision(1) << scale*hlAccept_[i] << " "
           << right << setw(7) << fixed << setprecision(1) <<
              ((hlAccept_[refIndex_]-hlAccept_[i] > 0) ? refRate_*ROOT::Math::beta_quantile(alpha, hlAccept_[i]+1, hlAccept_[refIndex_]-hlAccept_[i]) : 0) << " "
	   << right << setw(7) << hlErrors_[i] << " "
	   << hlNames_[i] << endl;
    }
  }

  LogVerbatim("HLTrigRprtTt") << endl;
  LogVerbatim("HLTrigRprtTt") << "HLT-Report " << "---------- HLTrig Summary ------------" << endl;
  LogVerbatim("HLTrigRprtTt") << "HLT-Report "
	 << right << setw(7) << "HLT #" << " "
	 << right << setw(7) << "WasRun" << " "
	 << right << setw(7) << "L1S" << " "
	 << right << setw(7) << "Pre" << " "
	 << right << setw(7) << "HLT" << " "
	 << right << setw(9) << "%L1sPre" << " "
         << right << setw(7) << "Rate" << " "
         << right << setw(7) << "RateHi" << " "
         << right << setw(7) << "HLTtot" << " "
         << right << setw(7) << "RateTot" << " "
	 << right << setw(7) << "Errors" << " "
	 << "Name" << endl;

  if (n>0) {
    for (unsigned int i=0; i!=n; ++i) {
      LogVerbatim("HLTrigRprtTt") << "HLT-Report "
	   << right << setw(7) << i << " "
	   << right << setw(7) << hlWasRun_[i] << " "
	   << right << setw(7) << hltL1s_[i] << " "
	   << right << setw(7) << hltPre_[i] << " "
	   << right << setw(7) << hlAccept_[i] << " "
	   << right << setw(9) << fixed << setprecision(5)
	   << static_cast<float>(100*hlAccept_[i])/
	      static_cast<float>(max(hltPre_[i], 1u)) << " "
           << right << setw(7) << fixed << setprecision(1) << scale*hlAccept_[i] << " "
           << right << setw(7) << fixed << setprecision(1) <<
              ((hlAccept_[refIndex_]-hlAccept_[i] > 0) ? refRate_*ROOT::Math::beta_quantile(alpha, hlAccept_[i]+1, hlAccept_[refIndex_]-hlAccept_[i]) : 0) << " "
           << right << setw(7) << hlAccTot_[i] << " "
           << right << setw(7) << fixed << setprecision(1) << scale*hlAccTot_[i] << " "
	   << right << setw(7) << hlErrors_[i] << " "
	   << hlNames_[i] << endl;
    }

    // now for each dataset
    for (size_t ds=0; ds<hlIndex_.size(); ++ds) {
      LogVerbatim("HLTrigRprtPD") << endl;
      LogVerbatim("HLTrigRprtPD") << "HLT-Report " << "---------- Dataset Summary: " << datasetNames_[ds] << " ------------" << hlAllTotDS_[ds] << endl;
      LogVerbatim("HLTrigRprtPD") << "HLT-Report "
         << right << setw(7) << "HLT #" << " "
         << right << setw(7) << "WasRun" << " "
         << right << setw(7) << "L1S" << " "
         << right << setw(7) << "Pre" << " "
         << right << setw(7) << "HLT" << " "
         << right << setw(9) << "%L1sPre" << " "
         << right << setw(7) << "Rate" << " "
         << right << setw(7) << "RateHi" << " "
         << right << setw(7) << "HLTtot" << " "
         << right << setw(7) << "RateTot" << " "
         << right << setw(7) << "Errors" << " "
         << "Name" << endl;
      for (size_t p=0; p<hlIndex_[ds].size(); ++p) {
        LogVerbatim("HLTrigRprtPD") << "HLT-Report "
           << right << setw(7) << p << " "
           << right << setw(7) << hlWasRun_[hlIndex_[ds][p]] << " "
           << right << setw(7) << hltL1s_[hlIndex_[ds][p]] << " "
           << right << setw(7) << hltPre_[hlIndex_[ds][p]] << " "
           << right << setw(7) << hlAccept_[hlIndex_[ds][p]] << " "
           << right << setw(9) << fixed << setprecision(5)
           << static_cast<float>(100*hlAccept_[hlIndex_[ds][p]])/
              static_cast<float>(max(hltPre_[hlIndex_[ds][p]], 1u)) << " "
           << right << setw(7) << fixed << setprecision(1) << scale*hlAccept_[hlIndex_[ds][p]] << " "
           << right << setw(7) << fixed << setprecision(1) <<
              ((hlAccept_[refIndex_]-hlAccept_[hlIndex_[ds][p]] > 0) ? refRate_*ROOT::Math::beta_quantile(alpha, hlAccept_[hlIndex_[ds][p]]+1, hlAccept_[refIndex_]-hlAccept_[hlIndex_[ds][p]]) : 0) << " "
           << right << setw(7) << hlAccTotDS_[ds][p] << " "
           << right << setw(7) << fixed << setprecision(1) << scale*hlAccTotDS_[ds][p] << " "
           << right << setw(7) << hlErrors_[hlIndex_[ds][p]] << " "
           << hlNames_[hlIndex_[ds][p]] << endl;
      }
    }

    // now for each stream
    for (size_t s=0; s<dsIndex_.size(); ++s) {
      LogVerbatim("HLTrigRprtST") << endl;
      LogVerbatim("HLTrigRprtST") << "HLT-Report " << "---------- Stream Summary: " << streamNames_[s] << " ------------" << dsAllTotS_[s] << endl;
      LogVerbatim("HLTrigRprtST") << "HLT-Report "
         << right << setw(10) << "Dataset #" << " "
         << right << setw(10) << "Individual" << " "
         << right << setw(10) << "Total" << " "
         << right << setw(10) << "Rate" << " "
         << right << setw(10) << "RateHi" << " "
         << right << setw(10) << "RateTot" << " "
         << "Name" << endl;
      for (size_t ds=0;ds<dsIndex_[s].size(); ++ds) {
        unsigned int acceptedDS = hlAccTotDS_[dsIndex_[s][ds]][hlIndex_[dsIndex_[s][ds]].size()-1];
        LogVerbatim("HLTrigRprtST") << "HLT-Report "
           << right << setw(10) << ds << " "
           << right << setw(10) << acceptedDS << " "
           << right << setw(10) << dsAccTotS_[s][ds] << " "
           << right << setw(10) << fixed << setprecision(1) << scale*acceptedDS << " "
           << right << setw(10) << fixed << setprecision(1) <<
              ((hlAccept_[refIndex_]-acceptedDS > 0) ? refRate_*ROOT::Math::beta_quantile(alpha, acceptedDS+1, hlAccept_[refIndex_]-acceptedDS) : 0) << " "
           << right << setw(10) << fixed << setprecision(1) << scale*dsAccTotS_[s][ds] << " "
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

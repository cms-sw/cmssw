// Note to self: the implementation uses TH1F's to store the L1T and HLT rates.
// Assuming a maximum rate of 100 kHz times a period of 23.31 s, one needs to store counts up to ~2.3e6.
// A "float" has 24 bits of precision, so it can store up to 2**24 ~ 16.7e6 without loss of precision.

// C++ headers
#include <algorithm>
#include <string>
#include <vector>

#include <fmt/printf.h>

// boost headers
#include <boost/regex.hpp>

// CMSSW headers
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

namespace {

  using dqm::reco::MonitorElement;

  struct RunBasedHistograms {
    // HLT configuration
    struct HLTIndices {
      unsigned int index_l1_seed;
      unsigned int index_prescale;

      HLTIndices() : index_l1_seed((unsigned int)-1), index_prescale((unsigned int)-1) {}
    };

    HLTConfigProvider hltConfig;
    std::vector<HLTIndices> hltIndices;

    std::vector<std::vector<unsigned int>> datasets;
    std::vector<std::vector<unsigned int>> streams;

    // L1T and HLT rate plots

    // per-path HLT plots
    struct HLTRatesPlots {
      MonitorElement *pass_l1_seed;
      MonitorElement *pass_prescale;
      MonitorElement *accept;
      MonitorElement *reject;
      MonitorElement *error;
    };

    // overall event count and event types
    MonitorElement *events_processed;
    std::vector<MonitorElement *> tcds_counts;

    // L1T triggers
    std::vector<MonitorElement *> l1t_counts;

    // HLT triggers
    std::vector<std::vector<HLTRatesPlots>> hlt_by_dataset_counts;

    // datasets
    std::vector<MonitorElement *> dataset_counts;

    // streams
    std::vector<MonitorElement *> stream_counts;

    RunBasedHistograms()
        :  // L1T and HLT configuration
          hltConfig(),
          hltIndices(),
          datasets(),
          streams(),
          // overall event count and event types
          events_processed(),
          tcds_counts(),
          // L1T triggers
          l1t_counts(),
          // HLT triggers
          hlt_by_dataset_counts(),
          // datasets
          dataset_counts(),
          // streams
          stream_counts() {}
  };
}  // namespace

class TriggerRatesMonitor : public DQMGlobalEDAnalyzer<RunBasedHistograms> {
public:
  explicit TriggerRatesMonitor(edm::ParameterSet const &);
  ~TriggerRatesMonitor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &, RunBasedHistograms &) const override;
  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      RunBasedHistograms &) const override;
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, RunBasedHistograms const &) const override;

  // TCDS trigger types
  // see https://twiki.cern.ch/twiki/bin/viewauth/CMS/TcdsEventRecord
  static constexpr const char *const s_tcds_trigger_types[] = {
      "Empty",          //  0 - No trigger
      "Physics",        //  1 - GT trigger
      "Calibration",    //  2 - Sequence trigger (calibration)
      "Random",         //  3 - Random trigger
      "Auxiliary",      //  4 - Auxiliary (CPM front panel NIM input) trigger
      nullptr,          //  5 - reserved
      nullptr,          //  6 - reserved
      nullptr,          //  7 - reserved
      "Cyclic",         //  8 - Cyclic trigger
      "Bunch-pattern",  //  9 - Bunch-pattern trigger
      "Software",       // 10 - Software trigger
      "TTS",            // 11 - TTS-sourced trigger
      nullptr,          // 12 - reserved
      nullptr,          // 13 - reserved
      nullptr,          // 14 - reserved
      nullptr           // 15 - reserved
  };

  // module configuration
  const edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> m_l1tMenuToken;
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> m_l1t_results;
  const edm::EDGetTokenT<edm::TriggerResults> m_hlt_results;
  const std::string m_dqm_path;
  const uint32_t m_lumisections_range;
};

// definition
constexpr const char *const TriggerRatesMonitor::s_tcds_trigger_types[];

void TriggerRatesMonitor::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("l1tResults", edm::InputTag("gtStage2Digis"));
  desc.addUntracked<edm::InputTag>("hltResults", edm::InputTag("TriggerResults"));
  desc.addUntracked<std::string>("dqmPath", "HLT/TriggerRates");
  desc.addUntracked<uint32_t>("lumisectionRange", 2500);  // ~16 hours
  descriptions.add("triggerRatesMonitor", desc);
}

TriggerRatesMonitor::TriggerRatesMonitor(edm::ParameterSet const &config)
    :  // module configuration
      m_l1tMenuToken{esConsumes<edm::Transition::BeginRun>()},
      m_l1t_results(consumes<GlobalAlgBlkBxCollection>(config.getUntrackedParameter<edm::InputTag>("l1tResults"))),
      m_hlt_results(consumes<edm::TriggerResults>(config.getUntrackedParameter<edm::InputTag>("hltResults"))),
      m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      m_lumisections_range(config.getUntrackedParameter<uint32_t>("lumisectionRange")) {}

void TriggerRatesMonitor::dqmBeginRun(edm::Run const &run,
                                      edm::EventSetup const &setup,
                                      RunBasedHistograms &histograms) const {
  histograms.tcds_counts.clear();
  histograms.tcds_counts.resize(sizeof(s_tcds_trigger_types) / sizeof(const char *));

  // cache the L1T trigger menu
  histograms.l1t_counts.clear();
  histograms.l1t_counts.resize(GlobalAlgBlk::maxPhysicsTriggers);

  // initialise the HLTConfigProvider
  bool changed = true;
  edm::EDConsumerBase::Labels labels;
  labelsForToken(m_hlt_results, labels);
  if (histograms.hltConfig.init(run, setup, labels.process, changed)) {
    // number of trigger paths in labels.process
    auto const nTriggers = histograms.hltConfig.size();
    histograms.hltIndices.resize(nTriggers);

    unsigned int const nDatasets = histograms.hltConfig.datasetNames().size();
    histograms.hlt_by_dataset_counts.clear();
    histograms.hlt_by_dataset_counts.resize(nDatasets);

    histograms.datasets.clear();
    histograms.datasets.resize(nDatasets);
    for (unsigned int i = 0; i < nDatasets; ++i) {
      auto const &paths = histograms.hltConfig.datasetContent(i);
      histograms.hlt_by_dataset_counts[i].resize(paths.size());
      histograms.datasets[i].reserve(paths.size());
      for (auto const &path : paths) {
        auto const triggerIdx = histograms.hltConfig.triggerIndex(path);
        if (triggerIdx < nTriggers)
          histograms.datasets[i].push_back(triggerIdx);
        else
          LogDebug("TriggerRatesMonitor")
              << "The rates of the HLT path \"" << path << "\" (dataset: \"" << histograms.hltConfig.datasetName(i)
              << "\") will not be monitored for this run.\nThis HLT path is not available in the process \""
              << labels.process << "\", but it is listed in its \"datasets\" PSet.";
      }
    }
    histograms.dataset_counts.clear();
    histograms.dataset_counts.resize(nDatasets);

    unsigned int const nStreams = histograms.hltConfig.streamNames().size();
    histograms.streams.clear();
    histograms.streams.resize(nStreams);
    for (unsigned int i = 0; i < nStreams; ++i) {
      for (auto const &dataset : histograms.hltConfig.streamContent(i)) {
        for (auto const &path : histograms.hltConfig.datasetContent(dataset)) {
          auto const triggerIdx = histograms.hltConfig.triggerIndex(path);
          if (triggerIdx < nTriggers)
            histograms.streams[i].push_back(triggerIdx);
          else
            LogDebug("TriggerRatesMonitor")
                << "The rates of the HLT path \"" << path << "\" (stream: \"" << histograms.hltConfig.streamName(i)
                << "\", dataset: \"" << dataset << "\") will not be monitored for this run.\n"
                << "This HLT path is not available in the process \"" << labels.process
                << "\", but it is listed in its \"datasets\" PSet.";
        }
      }
      std::sort(histograms.streams[i].begin(), histograms.streams[i].end());
      auto unique_end = std::unique(histograms.streams[i].begin(), histograms.streams[i].end());
      histograms.streams[i].resize(unique_end - histograms.streams[i].begin());
      histograms.streams[i].shrink_to_fit();
    }
    histograms.stream_counts.clear();
    histograms.stream_counts.resize(nStreams);
  } else {
    // HLTConfigProvider not initialised, skip the the HLT monitoring
    edm::LogError("TriggerRatesMonitor") << "Failed to initialise HLTConfigProvider: the rates of HLT triggers, "
                                            "datasets and streams will not be monitored for this run.";
  }
}

void TriggerRatesMonitor::bookHistograms(DQMStore::IBooker &booker,
                                         edm::Run const &run,
                                         edm::EventSetup const &setup,
                                         RunBasedHistograms &histograms) const {
  // book histograms for the overall event count, and trigger types
  booker.setCurrentFolder(m_dqm_path);
  histograms.events_processed = booker.book1D(
      "events", "Processed events vs. lumisection", m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);
  booker.setCurrentFolder(m_dqm_path + "/TCDS");
  unsigned int const sizeof_tcds_trigger_types = sizeof(s_tcds_trigger_types) / sizeof(const char *);
  if (sizeof_tcds_trigger_types == histograms.tcds_counts.size()) {
    for (unsigned int i = 0; i < sizeof_tcds_trigger_types; ++i)
      if (s_tcds_trigger_types[i]) {
        std::string const &title = fmt::sprintf("%s events vs. lumisection", s_tcds_trigger_types[i]);
        histograms.tcds_counts[i] =
            booker.book1D(s_tcds_trigger_types[i], title, m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);
      }
  } else
    edm::LogError("TriggerRatesMonitor")
        << "This should never happen: size of \"s_tcds_trigger_types\" array (" << sizeof_tcds_trigger_types
        << ") differs from size of \"histograms.tcds_counts\" vector (size=" << histograms.tcds_counts.size()
        << ").\nRate histograms of TCDS trigger types will not be booked for this run.";

  // book the rate histograms for the L1T triggers that are included in the L1T menu
  booker.setCurrentFolder(m_dqm_path + "/L1T");
  auto const &l1tMenu = setup.getData(m_l1tMenuToken);
  for (auto const &keyval : l1tMenu.getAlgorithmMap()) {
    unsigned int const bit = keyval.second.getIndex();
    if (bit >= histograms.l1t_counts.size()) {
      edm::LogError("TriggerRatesMonitor")
          << "This should never happen: bit of L1T algorithm (bit=" << bit << ", name=\"" << keyval.first
          << "\") is not smaller than size of \"histograms.l1t_counts\" vector (size=" << histograms.l1t_counts.size()
          << ").\nRate histogram of this L1T algorithm will not be booked for this run.";
      continue;
    }
    bool masked = false;  // FIXME read L1T masks once they will be avaiable in the EventSetup
    std::string const &name = fmt::sprintf("%s (bit %d)", keyval.first, bit);
    std::string const &title =
        fmt::sprintf("%s (bit %d)%s vs. lumisection", keyval.first, bit, (masked ? " (masked)" : ""));
    histograms.l1t_counts[bit] = booker.book1D(name, title, m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);
  }

  if (histograms.hltConfig.inited()) {
    auto const &datasetNames = histograms.hltConfig.datasetNames();

    // book the rate histograms for the HLT triggers
    for (unsigned int d = 0; d < datasetNames.size(); ++d) {
      booker.setCurrentFolder(m_dqm_path + "/HLT/" + datasetNames[d]);
      for (unsigned int i = 0; i < histograms.datasets[d].size(); ++i) {
        unsigned int index = histograms.datasets[d][i];
        std::string const &name = histograms.hltConfig.triggerName(index);
        histograms.hlt_by_dataset_counts[d][i].pass_l1_seed = booker.book1D(name + "_pass_L1_seed",
                                                                            name + " pass L1 seed, vs. lumisection",
                                                                            m_lumisections_range + 1,
                                                                            -0.5,
                                                                            m_lumisections_range + 0.5);
        histograms.hlt_by_dataset_counts[d][i].pass_prescale = booker.book1D(name + "_pass_prescaler",
                                                                             name + " pass prescaler, vs. lumisection",
                                                                             m_lumisections_range + 1,
                                                                             -0.5,
                                                                             m_lumisections_range + 0.5);
        histograms.hlt_by_dataset_counts[d][i].accept = booker.book1D(name + "_accept",
                                                                      name + " accept, vs. lumisection",
                                                                      m_lumisections_range + 1,
                                                                      -0.5,
                                                                      m_lumisections_range + 0.5);
        histograms.hlt_by_dataset_counts[d][i].reject = booker.book1D(name + "_reject",
                                                                      name + " reject, vs. lumisection",
                                                                      m_lumisections_range + 1,
                                                                      -0.5,
                                                                      m_lumisections_range + 0.5);
        histograms.hlt_by_dataset_counts[d][i].error = booker.book1D(name + "_error",
                                                                     name + " error, vs. lumisection",
                                                                     m_lumisections_range + 1,
                                                                     -0.5,
                                                                     m_lumisections_range + 0.5);
      }

      for (unsigned int i : histograms.datasets[d]) {
        // look for the index of the (last) L1T seed and prescale module in each path
        histograms.hltIndices[i].index_l1_seed = histograms.hltConfig.size(i);
        histograms.hltIndices[i].index_prescale = histograms.hltConfig.size(i);
        for (unsigned int j = 0; j < histograms.hltConfig.size(i); ++j) {
          std::string const &label = histograms.hltConfig.moduleLabel(i, j);
          std::string const &type = histograms.hltConfig.moduleType(label);
          if (type == "HLTL1TSeed" or type == "HLTLevel1GTSeed" or type == "HLTLevel1Activity" or
              type == "HLTLevel1Pattern") {
            // there might be more L1T seed filters in sequence
            // keep looking and store the index of the last one
            histograms.hltIndices[i].index_l1_seed = j;
          } else if (type == "HLTPrescaler") {
            // there should be only one prescaler in a path, and it should follow all L1T seed filters
            histograms.hltIndices[i].index_prescale = j;
            break;
          }
        }
      }
    }

    // book the rate histograms for the HLT datasets
    booker.setCurrentFolder(m_dqm_path + "/Datasets");
    for (unsigned int i = 0; i < datasetNames.size(); ++i)
      histograms.dataset_counts[i] =
          booker.book1D(datasetNames[i], datasetNames[i], m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);

    // book the rate histograms for the HLT streams
    booker.setCurrentFolder(m_dqm_path + "/Streams");
    auto const &streamNames = histograms.hltConfig.streamNames();
    for (unsigned int i = 0; i < streamNames.size(); ++i)
      histograms.stream_counts[i] =
          booker.book1D(streamNames[i], streamNames[i], m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);
  }
}

void TriggerRatesMonitor::dqmAnalyze(edm::Event const &event,
                                     edm::EventSetup const &setup,
                                     RunBasedHistograms const &histograms) const {
  unsigned int lumisection = event.luminosityBlock();

  // monitor the overall event count and event types rates
  histograms.events_processed->Fill(lumisection);
  if (histograms.tcds_counts[event.experimentType()])
    histograms.tcds_counts[event.experimentType()]->Fill(lumisection);

  // monitor the rates of L1T triggers
  auto const &bxvector = event.get(m_l1t_results);
  if (not bxvector.isEmpty(0)) {
    auto const &results = bxvector.at(0, 0);
    for (unsigned int i = 0; i < GlobalAlgBlk::maxPhysicsTriggers; ++i)
      if (results.getAlgoDecisionFinal(i))
        if (histograms.l1t_counts[i])
          histograms.l1t_counts[i]->Fill(lumisection);
  }

  // monitor the rates of HLT triggers, datasets and streams
  if (histograms.hltConfig.inited()) {
    auto const &hltResults = event.get(m_hlt_results);
    if (hltResults.size() != histograms.hltIndices.size()) {
      edm::LogError("TriggerRatesMonitor")
          << "This should never happen: the number of HLT paths has changed since the beginning of the run"
          << " (from " << histograms.hltIndices.size() << " to " << hltResults.size() << ").\n"
          << "Histograms for rates of HLT paths, datasets and streams will not be filled for this event.";
      return;
    }

    for (unsigned int d = 0; d < histograms.datasets.size(); ++d) {
      for (unsigned int i : histograms.datasets[d])
        if (hltResults[i].accept()) {
          histograms.dataset_counts[d]->Fill(lumisection);
          // ensure each dataset is incremented only once per event
          break;
        }
      for (unsigned int i = 0; i < histograms.datasets[d].size(); ++i) {
        unsigned int const index = histograms.datasets[d][i];
        edm::HLTPathStatus const &path = hltResults[index];

        if (path.index() > histograms.hltIndices[index].index_l1_seed)
          histograms.hlt_by_dataset_counts[d][i].pass_l1_seed->Fill(lumisection);
        if (path.index() > histograms.hltIndices[index].index_prescale)
          histograms.hlt_by_dataset_counts[d][i].pass_prescale->Fill(lumisection);
        if (path.accept())
          histograms.hlt_by_dataset_counts[d][i].accept->Fill(lumisection);
        else if (path.error())
          histograms.hlt_by_dataset_counts[d][i].error->Fill(lumisection);
        else
          histograms.hlt_by_dataset_counts[d][i].reject->Fill(lumisection);
      }
    }

    for (unsigned int i = 0; i < histograms.streams.size(); ++i)
      for (unsigned int j : histograms.streams[i])
        if (hltResults[j].accept()) {
          histograms.stream_counts[i]->Fill(lumisection);
          // ensure each stream is incremented only once per event
          break;
        }
  }
}

// define this as a plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TriggerRatesMonitor);

// Implementation based on FastTimerService.
// Target usage is to generate fake DAQ histograms for performance tests on RUBU machines (effect of running fastHadd) and to include in unit tests.

// CMSSW headers
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

namespace {

  struct RunBasedHistograms {
    // overall event count and event types
    dqm::reco::MonitorElement *events_processed;
    std::vector<dqm::reco::MonitorElement *> element_array;

    RunBasedHistograms()
        :  // overall event count and event types
          events_processed(nullptr),
          element_array() {}
  };
}  // namespace

class DaqTestHistograms : public DQMGlobalEDAnalyzer<RunBasedHistograms> {
public:
  explicit DaqTestHistograms(edm::ParameterSet const &);
  ~DaqTestHistograms() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void dqmBeginRun(edm::Run const &, edm::EventSetup const &, RunBasedHistograms &) const override;
  void bookHistograms(DQMStore::IBooker &,
                      edm::Run const &,
                      edm::EventSetup const &,
                      RunBasedHistograms &) const override;
  void dqmAnalyze(edm::Event const &, edm::EventSetup const &, RunBasedHistograms const &) const override;

  // module configuration
  const std::string m_dqm_path;
  const uint32_t m_lumisections_range;
  const uint32_t m_num_histograms;
};

void DaqTestHistograms::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("dqmPath", "DAQTEST/Test");
  desc.addUntracked<uint32_t>("lumisectionRange", 25);
  desc.addUntracked<uint32_t>("numberOfHistograms", 10);
  descriptions.add("dqmHLTTestMonitor", desc);
}

DaqTestHistograms::DaqTestHistograms(edm::ParameterSet const &config)
    :  // module configuration
      m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      m_lumisections_range(config.getUntrackedParameter<uint32_t>("lumisectionRange")),
      m_num_histograms(config.getUntrackedParameter<uint32_t>("numberOfHistograms")) {}

void DaqTestHistograms::dqmBeginRun(edm::Run const &run,
                                    edm::EventSetup const &setup,
                                    RunBasedHistograms &histograms) const {
  histograms.element_array.resize(m_num_histograms);
}

void DaqTestHistograms::bookHistograms(DQMStore::IBooker &booker,
                                       edm::Run const &run,
                                       edm::EventSetup const &setup,
                                       RunBasedHistograms &histograms) const {
  // book the overall event count and event types histograms
  booker.setCurrentFolder(m_dqm_path);
  histograms.events_processed = booker.book1D(
      "events", "Processed events vs. lumisection", m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);
  for (size_t i = 0; i < histograms.element_array.size(); i++) {
    std::stringstream strs;
    strs << "element " << i;
    std::stringstream strs2;
    strs2 << "e vs ls " << i;
    histograms.element_array[i] =
        booker.book1D(strs.str(), strs2.str(), m_lumisections_range + 1, -0.5, m_lumisections_range + 0.5);
  }
}

void DaqTestHistograms::dqmAnalyze(edm::Event const &event,
                                   edm::EventSetup const &setup,
                                   RunBasedHistograms const &histograms) const {
  unsigned int lumisection = event.luminosityBlock();

  histograms.events_processed->Fill(lumisection);
  for (size_t i = 0; i < histograms.element_array.size(); i++) {
    histograms.element_array[i]->Fill(lumisection);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DaqTestHistograms);

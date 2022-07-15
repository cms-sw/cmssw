// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/regex.hpp>

// Root headers
#include <TH1F.h>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

class ThroughputServiceClient : public DQMEDHarvester {
public:
  explicit ThroughputServiceClient(edm::ParameterSet const &);
  ~ThroughputServiceClient() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  const std::string m_dqm_path;
  const bool m_dqm_merge;

  void dqmEndLuminosityBlock(DQMStore::IBooker &booker,
                             DQMStore::IGetter &getter,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;
  void dqmEndJob(DQMStore::IBooker &booker, DQMStore::IGetter &getter) override;

private:
  void fillSummaryPlots(DQMStore::IBooker &booker, DQMStore::IGetter &getter);
};

ThroughputServiceClient::ThroughputServiceClient(edm::ParameterSet const &config)
    : m_dqm_path(config.getUntrackedParameter<std::string>("dqmPath")),
      m_dqm_merge(config.getUntrackedParameter<bool>("createSummary")) {}

void ThroughputServiceClient::dqmEndJob(DQMStore::IBooker &booker, DQMStore::IGetter &getter) {
  fillSummaryPlots(booker, getter);
}

void ThroughputServiceClient::dqmEndLuminosityBlock(DQMStore::IBooker &booker,
                                                    DQMStore::IGetter &getter,
                                                    edm::LuminosityBlock const &,
                                                    edm::EventSetup const &) {
  fillSummaryPlots(booker, getter);
}

void ThroughputServiceClient::fillSummaryPlots(DQMStore::IBooker &booker, DQMStore::IGetter &getter) {
  // find whether the plots are in the main folder, or in per-number-of-processess subfolders
  std::vector<std::string> folders;
  if (getter.get(m_dqm_path + "/throughput_sourced")) {
    // the plots are in the main folder
    folders.push_back(m_dqm_path);
  } else {
    static const boost::regex running_n_processes(".*/Running .*");
    booker.setCurrentFolder(m_dqm_path);
    std::vector<std::string> subdirs = getter.getSubdirs();
    for (auto const &subdir : subdirs) {
      if (boost::regex_match(subdir, running_n_processes)) {
        if (getter.get(subdir + "/throughput_sourced"))
          // the plots are in a per-number-of-processes subfolder
          folders.push_back(subdir);
      }
    }
  }
  // create a summary folder if there are more than one
  if (m_dqm_merge and folders.size() > 1) {
    std::string summary_folder = m_dqm_path + "/Summary";
    booker.setCurrentFolder(summary_folder);
    // clone the first set of histograms
    auto folder = folders.begin();
    TH1F *sourced =
        booker.book1D("throughput_sourced", getter.get(*folder + "/throughput_sourced")->getTH1F())->getTH1F();
    TH1F *retired =
        booker.book1D("throughput_retired", getter.get(*folder + "/throughput_retired")->getTH1F())->getTH1F();
    // add the other sets of histograms
    for (++folder; folder != folders.end(); ++folder) {
      sourced->Add(getter.get(*folder + "/throughput_sourced")->getTH1F());
      retired->Add(getter.get(*folder + "/throughput_retired")->getTH1F());
    }
    // move the summary folder to the list
    folders.push_back(std::move(summary_folder));
  }
  for (auto const &folder : folders) {
    TH1F *sourced = getter.get(folder + "/throughput_sourced")->getTH1F();
    TH1F *retired = getter.get(folder + "/throughput_retired")->getTH1F();
    booker.setCurrentFolder(folder);
    unsigned int nbins = sourced->GetXaxis()->GetNbins();
    double range = sourced->GetXaxis()->GetXmax();

    // (re)book and fill .../concurrent
    TH1F *concurrent = booker.book1D("concurrent", "Concurrent events being processed", nbins, 0., range)->getTH1F();
    double sum = 0;
    // from bin=0 (underflow) to bin=nbins+1 (overflow)
    for (unsigned int i = 0; i <= nbins + 1; ++i) {
      sum += sourced->GetBinContent(i) - retired->GetBinContent(i);
      concurrent->Fill(concurrent->GetXaxis()->GetBinCenter(i), sum);
    }

    TH1F *average = nullptr;
    double avg_min = std::min(sourced->GetMinimum(0.), retired->GetMinimum(0.));
    double avg_max = std::max(sourced->GetMaximum(), retired->GetMaximum());
    double width = avg_max - avg_min;
    avg_min = std::floor(avg_min - width * 0.2);
    if (avg_min < 0.)
      avg_min = 0.;
    avg_max = std::ceil(avg_max + width * 0.2);
    width = avg_max - avg_min;

    // define the range for .../average_sourced
    auto first = sourced->FindFirstBinAbove(0.);
    auto last = sourced->FindLastBinAbove(0.);
    booker.setCurrentFolder(folder);
    // (re)book and fill .../average_sourced
    average = booker.book1D("average_sourced", "Throughput (sourced events)", (int)width, avg_min, avg_max)->getTH1F();
    if (first >= 0)
      for (auto i = first; i <= last; ++i)
        average->Fill(sourced->GetBinContent(i));

    // define the range for .../average_retired
    first = retired->FindFirstBinAbove(0.);
    last = retired->FindLastBinAbove(0.);
    booker.setCurrentFolder(folder);
    // (re)book and fill .../average_retired
    average = booker.book1D("average_retired", "Throughput (retired events)", (int)width, avg_min, avg_max)->getTH1F();
    if (first >= 0)
      for (auto i = first; i <= last; ++i)
        average->Fill(retired->GetBinContent(i));
  }
}

void ThroughputServiceClient::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("dqmPath", "HLT/Throughput");
  desc.addUntracked<bool>("createSummary", true);
  descriptions.add("throughputServiceClient", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ThroughputServiceClient);

// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/regex.hpp>

// Root headers
#include <TH1F.h>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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
#include "DQMServices/Core/interface/MonitorElement.h"

class FastTimerServiceClient : public edm::EDAnalyzer {
public:
  explicit FastTimerServiceClient(edm::ParameterSet const &);
  ~FastTimerServiceClient();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  std::string m_dqm_path;

  void analyze(const edm::Event & event, const edm::EventSetup & setup) override;
  void endRun(edm::Run const & run, edm::EventSetup const & setup) override;
  void endLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup) override;

private:
  void fillSummaryPlots();
  void fillProcessSummaryPlots(std::string const & path);
  void fillPathSummaryPlots(double events, std::string const & path);
};


FastTimerServiceClient::FastTimerServiceClient(edm::ParameterSet const & config) :
  m_dqm_path( config.getUntrackedParameter<std::string>( "dqmPath" ) )
{
}

FastTimerServiceClient::~FastTimerServiceClient()
{
}

void
FastTimerServiceClient::analyze(edm::Event const & event, edm::EventSetup const & setup)
{
}

void
FastTimerServiceClient::endRun(edm::Run const & run, edm::EventSetup const & setup)
{
  fillSummaryPlots();
}

void
FastTimerServiceClient::endLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup)
{
  fillSummaryPlots();
}

void
FastTimerServiceClient::fillSummaryPlots(void)
{
  DQMStore * dqm = edm::Service<DQMStore>().operator->();
  if (dqm == 0)
    // cannot access the DQM store
    return;

  if (dqm->get(m_dqm_path + "/event")) {
    // the plots are directly in the configured folder
    fillProcessSummaryPlots(m_dqm_path);
  } else {
    static const boost::regex running_n_processes(".*/Running [0-9]+ processes");
    dqm->setCurrentFolder(m_dqm_path);
    std::vector<std::string> subdirs = dqm->getSubdirs();
    for (auto const & subdir: subdirs) {
      if (boost::regex_match(subdir, running_n_processes)) {
        // the plots are in a per-number-of-processes folder
        if (dqm->get(subdir + "/event"))
          fillProcessSummaryPlots(subdir);
      }
    }
  }
}



void
FastTimerServiceClient::fillProcessSummaryPlots(std::string const & current_path) {
  DQMStore * dqm = edm::Service<DQMStore>().operator->();

  MonitorElement * me = dqm->get(current_path + "/event");
  if (me == 0)
    // no FastTimerService DQM information
    return;

  double events = me->getTH1F()->GetEntries();

  // look for per-process directories
  static const boost::regex process_name(".*/process .*");
  dqm->setCurrentFolder(current_path);
  std::vector<std::string> subdirs = dqm->getSubdirs();
  for (auto const & subdir: subdirs) {
    if (boost::regex_match(subdir, process_name)) {
      // look for per-path plots inside each per-process directory
      if (dqm->dirExists(subdir + "/Paths"))
        fillPathSummaryPlots(events, subdir);
    }
  }

  // look for per-path plots inside the current directory
  if (dqm->dirExists(current_path + "/Paths"))
    fillPathSummaryPlots(events, current_path);
}

void
FastTimerServiceClient::fillPathSummaryPlots(double events, std::string const & current_path) {
  DQMStore * dqm = edm::Service<DQMStore>().operator->();

  // note: the following checks need to be kept separate, as any of these histograms might be missing
  // if any of them is filled, size will have the total number of paths, and "paths" can be used to extract the list of labels
  MonitorElement * me;
  TProfile const * paths = nullptr;
  uint32_t         size  = 0;

  // extract the list of Paths and EndPaths from the summary plots
  if (( me = dqm->get(current_path + "/paths_active_time") )) {
    paths = me->getTProfile();
    size  = paths->GetXaxis()->GetNbins();
  } else
  if (( me = dqm->get(current_path + "/paths_total_time") )) {
    paths = me->getTProfile();
    size  = paths->GetXaxis()->GetNbins();
  } else
  if (( me = dqm->get(current_path + "/paths_exclusive_time") )) {
    paths = me->getTProfile();
    size  = paths->GetXaxis()->GetNbins();
  }

  if (paths == nullptr)
    return;

  // for each path, fill histograms with
  //  - the average time spent in each module (total time spent in that module, averaged over all events)
  //  - the running time spent in each module (total time spent in that module, averaged over the events where that module actually ran)
  //  - the "efficiency" of each module (number of time a module succeded divided by the number of times the has run)
  dqm->setCurrentFolder(current_path + "/Paths");
  for (uint32_t p = 1; p <= size; ++p) {
    // extract the list of Paths and EndPaths from the bin labels of one of the summary plots
    std::string label = paths->GetXaxis()->GetBinLabel(p);
    MonitorElement * me_counter = dqm->get( current_path + "/Paths/" + label + "_module_counter" );
    MonitorElement * me_total   = dqm->get( current_path + "/Paths/" + label + "_module_total" );
    if (me_counter == 0 or me_total == 0)
      continue;
    TH1F * counter = me_counter->getTH1F();
    TH1F * total   = me_total  ->getTH1F();
    uint32_t bins = counter->GetXaxis()->GetNbins();
    double   min  = counter->GetXaxis()->GetXmin();
    double   max  = counter->GetXaxis()->GetXmax();
    TH1F * average    = dqm->book1D(label + "_module_average",    label + " module average",    bins, min, max)->getTH1F();
    average   ->SetYTitle("processing time [ms]");
    TH1F * running    = dqm->book1D(label + "_module_running",    label + " module running",    bins, min, max)->getTH1F();
    running   ->SetYTitle("processing time [ms]");
    TH1F * efficiency = dqm->book1D(label + "_module_efficiency", label + " module efficiency", bins, min, max)->getTH1F();
    efficiency->SetYTitle("filter efficiency");
    efficiency->SetMaximum(1.05);
    for (uint32_t i = 1; i <= bins; ++i) {
      const char * module = counter->GetXaxis()->GetBinLabel(i);
      average   ->GetXaxis()->SetBinLabel(i, module);
      running   ->GetXaxis()->SetBinLabel(i, module);
      efficiency->GetXaxis()->SetBinLabel(i, module);
      double t = total  ->GetBinContent(i);
      double n = counter->GetBinContent(i);
      double p = counter->GetBinContent(i+1);
      average   ->SetBinContent(i, t / events);
      if (n) {
        running   ->SetBinContent(i, t / n);
        efficiency->SetBinContent(i, p / n);
      }
    }
  }

}

void
FastTimerServiceClient::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>( "dqmPath", "HLT/TimerService");
  descriptions.add("fastTimerServiceClient", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FastTimerServiceClient);

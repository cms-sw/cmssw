// C++ headers
#include <string>
#include <cstring>

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

  void analyze(const edm::Event & event, const edm::EventSetup & setup);
  void beginJob();
  void endJob();
  void beginRun(edm::Run const & run, edm::EventSetup const & setup);
  void endRun  (edm::Run const & run, edm::EventSetup const & setup);
  void beginLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup);
  void endLuminosityBlock  (edm::LuminosityBlock const & lumi, edm::EventSetup const & setup);

private:
  void fillSummaryPlots();
};

FastTimerServiceClient::FastTimerServiceClient(edm::ParameterSet const & config) :
  m_dqm_path( config.getUntrackedParameter<std::string>( "dqmPath", "TimerService") )
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
FastTimerServiceClient::beginJob(void)
{
}

void
FastTimerServiceClient::endJob(void)
{
}

void
FastTimerServiceClient::beginRun(edm::Run const & run, edm::EventSetup const & setup)
{
}

void
FastTimerServiceClient::endRun(edm::Run const & run, edm::EventSetup const & setup)
{
  fillSummaryPlots();
}

void
FastTimerServiceClient::beginLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup)
{
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
  
  MonitorElement * me;
  me = dqm->get(m_dqm_path + "/event");
  if (me == 0)
    // no FastTimerService DQM information
    return;
  double events = me->getTH1F()->GetEntries();

  // note: the following (identical) loops need to be kept separate, as any of these group of histograms might be missing
  // if any of them is filled, size will have the total number of paths, and "paths" can be used to extract the list of labels
  dqm->setCurrentFolder(m_dqm_path);
  TH1F *   paths = nullptr;
  uint32_t size  = 0;

  // extract the list of Paths and EndPaths from the summary plots
  if (( me = dqm->get(m_dqm_path + "/paths_active_time") )) {
    paths = me->getTH1F();
    size  = paths->GetXaxis()->GetNbins();
  } else 
  if (( me = dqm->get(m_dqm_path + "/paths_total_time") )) {
    paths = me->getTH1F();
    size  = paths->GetXaxis()->GetNbins();
  } else
  if (( me = dqm->get(m_dqm_path + "/paths_exclusive_time") )) {
    paths = me->getTH1F();
    size  = paths->GetXaxis()->GetNbins();
  }

  // for each path, fill histograms with
  //  - the average time spent in each module (total time spent in that module, averaged over all events)
  //  - the running time spent in each module (total time spent in that module, averaged over the events where that module actually ran)
  //  - the "efficiency" of each module (number of time a module succeded divided by the number of times the has run)
  dqm->setCurrentFolder(m_dqm_path + "/Paths");
  for (uint32_t p = 1; p <= size; ++p) {
    // extract the list of Paths and EndPaths from the bin labels of one of the summary plots
    std::string label = paths->GetXaxis()->GetBinLabel(p);
    MonitorElement * me_counter = dqm->get( m_dqm_path + "/Paths/" + label + "_module_counter" );
    MonitorElement * me_total   = dqm->get( m_dqm_path + "/Paths/" + label + "_module_total" );
    if (me_counter == 0 or me_total == 0)
      continue;
    TH1F * counter = me_counter->getTH1F();
    TH1F * total   = me_total  ->getTH1F();
    uint32_t bins = counter->GetXaxis()->GetNbins();
    double   min  = counter->GetXaxis()->GetXmin();
    double   max  = counter->GetXaxis()->GetXmax();
    TH1F * average    = dqm->book1D(label + "_module_average",    label + " module average",    bins, min, max)->getTH1F();
    TH1F * running    = dqm->book1D(label + "_module_running",    label + " module running",    bins, min, max)->getTH1F();
    TH1F * efficiency = dqm->book1D(label + "_module_efficiency", label + " module efficiency", bins, min, max)->getTH1F();
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
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FastTimerServiceClient);

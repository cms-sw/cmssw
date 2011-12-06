// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/foreach.hpp>
// for forward compatibility with boost 1.47
#define BOOST_FILESYSTEM_VERSION 3
#include <boost/filesystem/path.hpp>

// Root headers
#include <TH1F.h>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class FastTimerServiceClient : public edm::EDAnalyzer {
public:
  explicit FastTimerServiceClient(edm::ParameterSet const &);
  ~FastTimerServiceClient();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  boost::filesystem::path   m_dqm_path;
  std::string               m_hlt_name;

  void analyze(const edm::Event & event, const edm::EventSetup & setup);                            // this must be implemented
  void beginJob();
  void endJob();
  void beginRun(edm::Run const & run, edm::EventSetup const & setup);
  void endRun(  edm::Run const & run, edm::EventSetup const & setup);
  void beginLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup);
  void endLuminosityBlock(  edm::LuminosityBlock const & lumi, edm::EventSetup const & setup);
};

FastTimerServiceClient::FastTimerServiceClient(edm::ParameterSet const & config) :
  m_dqm_path( config.getUntrackedParameter<std::string>( "dqmPath",         "TimerService") ),
  m_hlt_name( config.getUntrackedParameter<std::string>( "hltProcessName",  "HLT") )                // XXX unused
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
  DQMStore * dqm = edm::Service<DQMStore>().operator->();
  if (dqm == 0)
    // cannot access the DQM store
    return;
  
  MonitorElement * me;
  me = dqm->get( (m_dqm_path / "events").generic_string() );
  if (me == 0)
    // no FastTimerService DQM information
    return;
  double events = me->getTH1F()->GetEntries();

  // access the trigger configuration for the *current job*
  edm::service::TriggerNamesService & tns = * edm::Service<edm::service::TriggerNamesService>();
  std::vector<std::string> const & paths = tns.getTrigPaths();
  std::vector<std::string> const & endps = tns.getEndPaths();
  size_t size_p = paths.size();
  size_t size_e = endps.size();
  size_t size = size_p + size_e;

  // fill summary histograms with the average (total and active) time spent in each path
  dqm->setCurrentFolder(m_dqm_path.generic_string());
  TH1F * path_active = dqm->book1D("path_active_time", "Additional time spent in each path", size, -0.5, size-0.5)->getTH1F();
  TH1F * path_total  = dqm->book1D("path_total_time",  "Total time spent in each path", size, -0.5, size-0.5)->getTH1F();
  for (size_t i = 0; i < size_p; ++i) {
    std::string const & label = paths[i];
    path_active->GetXaxis()->SetBinLabel( i + 1, label.c_str() );
    path_total ->GetXaxis()->SetBinLabel( i + 1, label.c_str() );
    if (( me = dqm->get( (m_dqm_path / "Paths" / (label + "_total")).generic_string() ) ))
      path_total ->Fill(i, me->getTH1F()->GetMean());
    if (( me = dqm->get( (m_dqm_path / "Paths" / (label + "_active")).generic_string() ) ))
      path_active->Fill(i, me->getTH1F()->GetMean());

  }
  for (size_t i = 0; i < size_e; ++i) {
    std::string const & label = endps[i];
    path_active->GetXaxis()->SetBinLabel( i + 1, label.c_str() );
    path_total ->GetXaxis()->SetBinLabel( i + 1, label.c_str() );
    if (( me = dqm->get( (m_dqm_path / "Paths" / (label + "_total")).generic_string() ) ))
      path_total ->Fill(i, me->getTH1F()->GetMean());
    if (( me = dqm->get( (m_dqm_path / "Paths" / (label + "_active")).generic_string() ) ))
      path_active->Fill(i, me->getTH1F()->GetMean());
  }

  // for each path, fill histograms with
  //  - the average time spent in each module (total time spent in that module, averaged over all events)
  //  - the running time spent in each module (total time spent in that module, averaged over the events where that module actually ran)
  //  - the "efficiency" of each module (number of time a module succeded divided by the number of times the has run)
  dqm->setCurrentFolder((m_dqm_path / "Paths").generic_string());
  BOOST_FOREACH(std::string const & label, paths) {
    TH1F * counter = dqm->get( (m_dqm_path / "Paths" / (label+"_module_counter")).generic_string() )->getTH1F();
    TH1F * total   = dqm->get( (m_dqm_path / "Paths" / (label+"_module_total"  )).generic_string() )->getTH1F();
    if (counter == 0 or total == 0)
      continue;
    size_t bins = counter->GetXaxis()->GetNbins();
    double min  = counter->GetXaxis()->GetXmin();
    double max  = counter->GetXaxis()->GetXmax();
    TH1F * average    = dqm->book1D(label + "_module_average",    label + " module average",    bins, min, max)->getTH1F();
    TH1F * running    = dqm->book1D(label + "_module_running",    label + " module running",    bins, min, max)->getTH1F();
    TH1F * efficiency = dqm->book1D(label + "_module_efficiency", label + " module efficiency", bins, min, max)->getTH1F();
    for (size_t i = 1; i <= bins; ++i) {
      const char * module = counter->GetXaxis()->GetBinLabel(i);
      average   ->GetXaxis()->SetBinLabel(i, module);
      running   ->GetXaxis()->SetBinLabel(i, module);
      efficiency->GetXaxis()->SetBinLabel(i, module);
      double x = total  ->GetBinContent(i);
      double n = counter->GetBinContent(i);
      double p = counter->GetBinContent(i+1);
      average   ->SetBinContent(i, x / events);
      if (n) {
        running   ->SetBinContent(i, x / n);
        efficiency->SetBinContent(i, p / n);
      }
    }
  }
  // XXX move this to a function and call it twice, instead of duplicating the code
  BOOST_FOREACH(std::string const & label, endps) {
    TH1F * counter = dqm->get( (m_dqm_path / "Paths" / (label+"_module_counter")).generic_string() )->getTH1F();
    TH1F * total   = dqm->get( (m_dqm_path / "Paths" / (label+"_module_total"  )).generic_string() )->getTH1F();
    if (counter == 0 or total == 0)
      continue;
    size_t bins = counter->GetXaxis()->GetNbins();
    double min  = counter->GetXaxis()->GetXmin();
    double max  = counter->GetXaxis()->GetXmax();
    TH1F * average    = dqm->book1D(label + "_module_average",    label + " module average",    bins, min, max)->getTH1F();
    TH1F * running    = dqm->book1D(label + "_module_running",    label + " module running",    bins, min, max)->getTH1F();
    TH1F * efficiency = dqm->book1D(label + "_module_efficiency", label + " module efficiency", bins, min, max)->getTH1F();
    for (size_t i = 1; i <= bins; ++i) {
      const char * module = counter->GetXaxis()->GetBinLabel(i);
      average   ->GetXaxis()->SetBinLabel(i, module);
      running   ->GetXaxis()->SetBinLabel(i, module);
      efficiency->GetXaxis()->SetBinLabel(i, module);
      double x = total  ->GetBinContent(i);
      double n = counter->GetBinContent(i);
      double p = counter->GetBinContent(i+1);
      average   ->SetBinContent(i, x / events);
      if (n) {
        running   ->SetBinContent(i, x / n);
        efficiency->SetBinContent(i, p / n);
      }
    }
  }

}

void
FastTimerServiceClient::beginLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup)
{
}

void
FastTimerServiceClient::endLuminosityBlock(edm::LuminosityBlock const & lumi, edm::EventSetup const & setup)
{
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

// C++ headers
#include <string>
#include <cstring>

// boost headers
#include <boost/foreach.hpp>

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

  // fill summary histograms with the average (total and active) time spent in each path
  dqm->setCurrentFolder(m_dqm_path);
  TH1F * path_active = dqm->get(m_dqm_path + "/path_active_time")->getTH1F();
  TH1F * path_total  = dqm->get(m_dqm_path + "/path_total_time")->getTH1F();
  size_t size = path_total->GetXaxis()->GetNbins();
  for (size_t i = 0; i < size; ++i) {
    // extract the list of Paths and EndPaths from the bin labels of "path_total_time"
    std::string label = path_total->GetXaxis()->GetBinLabel(i+1);   // bin count from 1 (bin 0 is underflow)
    if (( me = dqm->get(m_dqm_path + "/Paths/" + label + "_total") ))
      path_total ->Fill(i, me->getTH1F()->GetMean());
    if (( me = dqm->get(m_dqm_path + "/Paths/" + label + "_active") ))
      path_active->Fill(i, me->getTH1F()->GetMean());
  }

  // for each path, fill histograms with
  //  - the average time spent in each module (total time spent in that module, averaged over all events)
  //  - the running time spent in each module (total time spent in that module, averaged over the events where that module actually ran)
  //  - the "efficiency" of each module (number of time a module succeded divided by the number of times the has run)
  dqm->setCurrentFolder(m_dqm_path + "/Paths");
  for (size_t p = 1; p <= size; ++p) {
    // extract the list of Paths and EndPaths from the bin labels of "path_total_time"
    std::string label = path_total->GetXaxis()->GetBinLabel(p);
    TH1F * counter = dqm->get( m_dqm_path + "/Paths/" + label + "_module_counter" )->getTH1F();
    TH1F * total   = dqm->get( m_dqm_path + "/Paths/" + label + "_module_total"   )->getTH1F();
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

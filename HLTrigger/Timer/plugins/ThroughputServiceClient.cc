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
#include "DQMServices/Core/interface/MonitorElement.h"

class ThroughputServiceClient : public DQMEDHarvester {
public:
  explicit ThroughputServiceClient(edm::ParameterSet const &);
  ~ThroughputServiceClient();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  std::string m_dqm_path;

  void dqmEndLuminosityBlock(DQMStore::IGetter & getter, edm::LuminosityBlock const & lumi, edm::EventSetup const & setup);
  void dqmEndJob(DQMStore::IBooker & booker, DQMStore::IGetter & getter);

private:
  void fillSummaryPlots(        DQMStore::IBooker & booker, DQMStore::IGetter & getter);
};


ThroughputServiceClient::ThroughputServiceClient(edm::ParameterSet const & config) :
  m_dqm_path( config.getUntrackedParameter<std::string>( "dqmPath" ) )
{
}

ThroughputServiceClient::~ThroughputServiceClient()
{
}

void
ThroughputServiceClient::dqmEndJob(DQMStore::IBooker & booker, DQMStore::IGetter & getter)
{
  fillSummaryPlots(booker, getter);
}

void
ThroughputServiceClient::dqmEndLuminosityBlock(DQMStore::IGetter & getter, edm::LuminosityBlock const & lumi, edm::EventSetup const & setup)
{
  // fillSummaryPlots(getter);
}

void
ThroughputServiceClient::fillSummaryPlots(DQMStore::IBooker & booker, DQMStore::IGetter & getter)
{
  // find whether the plots are in the main folder, or in per-number-of-processess subfolders
  std::vector<std::string> folders;
  if (getter.get(m_dqm_path + "/throughput_sourced")) {
    // the plots are in the main folder
    folders.push_back(m_dqm_path);
  } else {
    static const boost::regex running_n_processes(".*/Running [0-9]+ processes");
    booker.setCurrentFolder(m_dqm_path);
    std::vector<std::string> subdirs = getter.getSubdirs();
    for (auto const & subdir: subdirs) {
      if (boost::regex_match(subdir, running_n_processes)) {
        if (getter.get(subdir + "/throughput_sourced"))
          // the plots are in a per-number-of-processes subfolder
          folders.push_back(subdir + "/throughput_sourced");
      }
    }
  }
  for (auto const & folder: folders) {
    TH1F * sourced = getter.get( folder + "/throughput_sourced" )->getTH1F();
    TH1F * retired = getter.get( folder + "/throughput_retired" )->getTH1F();
    booker.setCurrentFolder(folder);
    unsigned int nbins = sourced->GetXaxis()->GetNbins();
    double       range = sourced->GetXaxis()->GetXmax();
    TH1F * concurrent = booker.book1D("concurrent", "Concurrent events being processed", nbins, 0., range)->getTH1F();
    double sum = 0;
    // from bin=0 (underflow) to bin=nbins+1 (overflow)
    for (unsigned int i = 0; i <= nbins+1; ++i) {
      sum += sourced->GetBinContent(i) - retired->GetBinContent(i);
      concurrent->Fill( concurrent->GetXaxis()->GetBinCenter(i), sum );
    }
  }
}

void
ThroughputServiceClient::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>( "dqmPath", "HLT/Throughput" );
  descriptions.add("throughputServiceClient", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ThroughputServiceClient);

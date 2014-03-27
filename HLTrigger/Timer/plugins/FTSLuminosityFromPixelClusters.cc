// C++ headers
#include <string>
#include <cstring>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class FTSLuminosityFromPixelClusters : public edm::global::EDAnalyzer<> {
public:
  explicit FTSLuminosityFromPixelClusters(edm::ParameterSet const &);
  ~FTSLuminosityFromPixelClusters();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::EDGetTokenT<SiPixelClusterCollectionNew> m_token;
  unsigned int                                  m_lumi_id;

  void analyze(edm::StreamID sid, edm::Event const & event, const edm::EventSetup & setup) const override;
};

FTSLuminosityFromPixelClusters::FTSLuminosityFromPixelClusters(edm::ParameterSet const & config) :
  m_token(consumes<SiPixelClusterCollectionNew>(config.getParameter<edm::InputTag>("source"))),
  m_lumi_id((unsigned int) -1)
{
  if (not edm::Service<FastTimerService>().isAvailable())
    return;

  std::string const & name  = config.getParameter<std::string>("name");
  std::string const & title = config.getParameter<std::string>("title");
  std::string const & label = config.getParameter<std::string>("label");
  double range              = config.getParameter<double>("range");
  double resolution         = config.getParameter<double>("resolution");

  m_lumi_id = edm::Service<FastTimerService>()->reserveLuminosityPlots(name, title, label, range, resolution);
}

FTSLuminosityFromPixelClusters::~FTSLuminosityFromPixelClusters()
{
}

void
FTSLuminosityFromPixelClusters::analyze(edm::StreamID sid, edm::Event const & event, edm::EventSetup const & setup) const
{
  if (not edm::Service<FastTimerService>().isAvailable())
    return;

  double value = 0.;
  edm::Handle<SiPixelClusterCollectionNew> h_clusters;
  if (event.getByToken(m_token, h_clusters))
        value = h_clusters->dataSize();

  edm::Service<FastTimerService>()->setLuminosity(sid, m_lumi_id, value);
}

void
FTSLuminosityFromPixelClusters::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("siPixelClusters"));
  desc.add<std::string>("name",  "clusters");
  desc.add<std::string>("title", "pixel clusters");
  desc.add<std::string>("label", "pixel clusters");
  desc.add<double>("range",      20000);
  desc.add<double>("resolution", 500);
  descriptions.add("ftsLuminosityFromPixelClusters", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FTSLuminosityFromPixelClusters);

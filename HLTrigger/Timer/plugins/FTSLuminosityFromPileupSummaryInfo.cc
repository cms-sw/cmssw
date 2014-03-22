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
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

class FTSLuminosityFromPileupSummaryInfo : public edm::global::EDAnalyzer<> {
public:
  explicit FTSLuminosityFromPileupSummaryInfo(edm::ParameterSet const &);
  ~FTSLuminosityFromPileupSummaryInfo();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  m_token;
  unsigned int                                      m_lumi_id;

  void analyze(edm::StreamID sid, edm::Event const & event, const edm::EventSetup & setup) const override;
};

FTSLuminosityFromPileupSummaryInfo::FTSLuminosityFromPileupSummaryInfo(edm::ParameterSet const & config) :
  m_token(consumes<std::vector<PileupSummaryInfo>>(config.getParameter<edm::InputTag>("source"))),
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

FTSLuminosityFromPileupSummaryInfo::~FTSLuminosityFromPileupSummaryInfo()
{
}

void
FTSLuminosityFromPileupSummaryInfo::analyze(edm::StreamID sid, edm::Event const & event, edm::EventSetup const & setup) const
{
  if (not edm::Service<FastTimerService>().isAvailable())
    return;

  double value = 0.;
  edm::Handle<std::vector<PileupSummaryInfo>> h_summary;
  if (event.getByToken(m_token, h_summary)) {
    for (PileupSummaryInfo const & pileup: *h_summary) {
      // only use the in-time pileup
      if (pileup.getBunchCrossing() == 0) {
        // use the per-event in-time pileup
        value = pileup.getPU_NumInteractions();
        // use the average pileup
        // value = pileup.getTrueNumInteractions();
        break;
      }
    }
  }

  edm::Service<FastTimerService>()->setLuminosity(sid, m_lumi_id, value);
}

void
FTSLuminosityFromPileupSummaryInfo::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("addPileupInfo"));
  desc.add<std::string>("name",  "pileup");
  desc.add<std::string>("title", "in-time pileup");
  desc.add<std::string>("label", "in-time pileup");
  desc.add<double>("range",      40);
  desc.add<double>("resolution", 1);
  descriptions.add("ftsLuminosityFromPileupSummaryInfo", desc);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FTSLuminosityFromPileupSummaryInfo);

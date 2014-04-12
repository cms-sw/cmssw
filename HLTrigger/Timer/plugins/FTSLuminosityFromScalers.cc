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
#include "DataFormats/Scalers/interface/LumiScalers.h"

class FTSLuminosityFromScalers : public edm::global::EDAnalyzer<> {
public:
  explicit FTSLuminosityFromScalers(edm::ParameterSet const &);
  ~FTSLuminosityFromScalers();

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  enum class Type {
    InstantaneousLuminosity,
    Pileup,
    Invalid = -1
  };

  static Type parse(std::string const & type) {
    if (type == "InstantaneousLuminosity")
      return Type::InstantaneousLuminosity;
    else if (type == "Pileup")
      return Type::Pileup;
    else
      return Type::Invalid;
  }

  edm::EDGetTokenT<LumiScalersCollection>   m_token;
  Type                                      m_type;
  unsigned int                              m_lumi_id;

  void analyze(edm::StreamID sid, edm::Event const & event, const edm::EventSetup & setup) const override;
};

FTSLuminosityFromScalers::FTSLuminosityFromScalers(edm::ParameterSet const & config) :
  m_token(consumes<LumiScalersCollection>(config.getParameter<edm::InputTag>("source"))),
  m_type(parse(config.getParameter<std::string>("type"))),
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

FTSLuminosityFromScalers::~FTSLuminosityFromScalers()
{
}

void
FTSLuminosityFromScalers::analyze(edm::StreamID sid, edm::Event const & event, edm::EventSetup const & setup) const
{
  if (not edm::Service<FastTimerService>().isAvailable())
    return;

  double value = 0.;
  edm::Handle<LumiScalersCollection> h_luminosity;
  if (event.getByToken(m_token, h_luminosity) and not h_luminosity->empty()) {
    switch (m_type) {
      case Type::InstantaneousLuminosity:
        value = h_luminosity->front().instantLumi() * 1.e30;
        break;
      case Type::Pileup:
        value = h_luminosity->front().pileup();
        break;
      case Type::Invalid:
        value = 0.;
        break;
    }
  }

  edm::Service<FastTimerService>()->setLuminosity(sid, m_lumi_id, value);
}

void
FTSLuminosityFromScalers::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  // instantaneous luminosity
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source", edm::InputTag("scalersRawToDigi"));
    desc.add<std::string>("type",  "InstantaneousLuminosity");
    desc.add<std::string>("name",  "luminosity");
    desc.add<std::string>("title", "instantaneous luminosity");
    desc.add<std::string>("label", "instantaneous luminosity [cm^{-2}s^{-1}]");
    desc.add<double>("range",      8.e33);
    desc.add<double>("resolution", 1.e31);
    descriptions.add("ftsLuminosityFromScalers", desc);
  }
  // pileup
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source", edm::InputTag("scalersRawToDigi"));
    desc.add<std::string>("type",  "Pileup");
    desc.add<std::string>("name",  "pileup");
    desc.add<std::string>("title", "pileup");
    desc.add<std::string>("label", "pileup");
    desc.add<double>("range",      40);
    desc.add<double>("resolution",  1);
    descriptions.add("ftsPileupFromScalers", desc);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FTSLuminosityFromScalers);

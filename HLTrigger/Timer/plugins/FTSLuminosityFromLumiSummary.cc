// C++ headers
#include <string>
#include <cstring>

// CMSSW headers
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/Timer/interface/FastTimerService.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "RecoLuminosity/LumiProducer/interface/LumiCorrectionParam.h"

class FTSLuminosityFromLumiSummary : public edm::global::EDAnalyzer<> {
public:
  explicit FTSLuminosityFromLumiSummary(edm::ParameterSet const &);
  ~FTSLuminosityFromLumiSummary();

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

  edm::EDGetTokenT<LumiSummary> m_token;
  double                        m_cross_section;
  Type                          m_type;
  unsigned int                  m_lumi_id;
  std::vector<double>           m_value;            // values are per-stream, computed in the StreamBeginLuminosityBlock and used in each event

  virtual void preallocStreams(unsigned int size) override;
  virtual void doStreamBeginLuminosityBlock_(edm::StreamID id, edm::LuminosityBlock const & lumi, edm::EventSetup const & setup) override;
  virtual void analyze(edm::StreamID sid, edm::Event const & event, const edm::EventSetup & setup) const override;
};

FTSLuminosityFromLumiSummary::FTSLuminosityFromLumiSummary(edm::ParameterSet const & config) :
  m_token(consumes<LumiSummary, edm::InLumi>(config.getParameter<edm::InputTag>("source"))),
  m_cross_section(config.getParameter<double>("crossSection")),
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

FTSLuminosityFromLumiSummary::~FTSLuminosityFromLumiSummary()
{
}

void FTSLuminosityFromLumiSummary::preallocStreams(unsigned int size)
{
  m_value.resize(size, 0.);
}

void FTSLuminosityFromLumiSummary::doStreamBeginLuminosityBlock_(edm::StreamID sid, edm::LuminosityBlock const & lumi, edm::EventSetup const & setup)
{
  m_value[sid] = 0.;

  edm::ESHandle<LumiCorrectionParam> corrector;
  setup.getData(corrector);
  if (not corrector.isValid()) {
    edm::LogError("FTSLuminosityFromLumiSummary") << "unable to calibrate the raw luminosity values, please add a LumiCorrectionSource ESProducer to your configuration";
    return;
  }

  edm::Handle<LumiSummary> h_summary;
  if (lumi.getByToken(m_token, h_summary)) {
    double correction = corrector->getCorrection(h_summary->avgInsDelLumi());
    /*
    std::cerr << "LumiSummary loaded" << std::endl;
    std::cerr << "  uncorrected luminosity: " << h_summary->avgInsDelLumi() << std::endl;
    std::cerr << "  correction factor:      " << correction << std::endl;
    std::cerr << "  corrected luminosity:   " << h_summary->avgInsDelLumi() * correction * 1.e30 << std::endl;
    std::cerr << "  integrated luminosity:  " << h_summary->intgDelLumi()   * correction * 1.e30 << std::endl;
    std::cerr << "  colliding bunches:      " << corrector->ncollidingbunches() << std::endl;
    std::cerr << "  orbits:                 " << h_summary->numOrbit() << std::endl;
    std::cerr << "  pileup:                 " << h_summary->intgDelLumi() * correction * m_cross_section * 1.e3 / h_summary->numOrbit() / corrector->ncollidingbunches() << std::endl;
    */
    switch (m_type) {
      case Type::InstantaneousLuminosity:
        m_value[sid] = h_summary->avgInsDelLumi() * correction * 1.e30;
        break;
      case Type::Pileup:
        // integrated luminosity [nb-1] * pp cross section [mb] * 10^6 nb/mb / 2^18 orbits / number of colliding bunches
        m_value[sid] = h_summary->intgDelLumi() * correction * m_cross_section * 1.e3 / h_summary->numOrbit() / corrector->ncollidingbunches();
        break;
      case Type::Invalid:
        m_value[sid] = 0.;
        break;
    }
  }
}

void
FTSLuminosityFromLumiSummary::analyze(edm::StreamID sid, edm::Event const & event, edm::EventSetup const & setup) const
{
  if (not edm::Service<FastTimerService>().isAvailable())
    return;

  edm::Service<FastTimerService>()->setLuminosity(sid, m_lumi_id, m_value[sid]);
}

void
FTSLuminosityFromLumiSummary::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  // instantaneous luminosity
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source", edm::InputTag("lumiProducer"));
    desc.add<double>("crossSection", 69.3);
    desc.add<std::string>("type",  "InstantaneousLuminosity");
    desc.add<std::string>("name",  "luminosity");
    desc.add<std::string>("title", "instantaneous luminosity");
    desc.add<std::string>("label", "instantaneous luminosity [cm^{-2}s^{-1}]");
    desc.add<double>("range",      8.e33);
    desc.add<double>("resolution", 1.e31);
    descriptions.add("ftsLuminosityFromLumiSummary", desc);
  }
  // pileup
  {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source", edm::InputTag("lumiProducer"));
    desc.add<double>("crossSection", 69.3);
    desc.add<std::string>("type",  "Pileup");
    desc.add<std::string>("name",  "pileup");
    desc.add<std::string>("title", "pileup");
    desc.add<std::string>("label", "pileup");
    desc.add<double>("range",      40);
    desc.add<double>("resolution",  1);
    descriptions.add("ftsPileupFromLumiSummary", desc);
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(FTSLuminosityFromLumiSummary);

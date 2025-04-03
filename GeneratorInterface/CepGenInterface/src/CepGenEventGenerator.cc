// CepGen-CMSSW interfacing module
//   2022-2024, Laurent Forthomme

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "GeneratorInterface/CepGenInterface/interface/CepGenEventGenerator.h"
#include "GeneratorInterface/CepGenInterface/interface/CepGenParametersConverter.h"

#include <CepGen/Core/Exception.h>
#include <CepGen/Core/RunParameters.h>
#include <CepGen/Event/Event.h>
#include <CepGen/EventFilter/EventExporter.h>
#include <CepGen/EventFilter/EventModifier.h>
#include <CepGen/Generator.h>
#include <CepGen/Modules/EventExporterFactory.h>
#include <CepGen/Modules/EventModifierFactory.h>
#include <CepGen/Modules/ProcessFactory.h>
#include <CepGen/Process/Process.h>
#include <CepGenAddOns/HepMC2Wrapper/HepMC2EventInterface.h>

using namespace gen;

CepGenEventGenerator::CepGenEventGenerator(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iC)
    : BaseHadronizer(iConfig),
      proc_params_(cepgen::fromParameterSet(iConfig.getParameter<edm::ParameterSet>("process"))) {
  // specify the overall module verbosity
  cepgen::utils::Logger::get().setLevel(
      static_cast<cepgen::utils::Logger::Level>(iConfig.getUntrackedParameter<int>("verbosity", 0)));

  // build the process
  edm::LogInfo("CepGenEventGenerator") << "Process to be generated: " << proc_params_ << ".";

  const auto modif_mods = cepgen::fromParameterSet(
      iConfig.getUntrackedParameter<edm::ParameterSet>("modifierModules", edm::ParameterSet{}));
  edm::LogInfo("CepGenEventGenerator") << "Event modifier modules: " << modif_mods << ".";
  for (const auto& mod : modif_mods.keys())
    modif_modules_.emplace_back(std::make_pair(mod, modif_mods.get<cepgen::ParametersList>(mod)));

  const auto output_mods =
      cepgen::fromParameterSet(iConfig.getUntrackedParameter<edm::ParameterSet>("outputModules", edm::ParameterSet{}));
  edm::LogInfo("CepGenEventGenerator") << "Output modules: " << output_mods << ".";
  for (const auto& mod : output_mods.keys())
    output_modules_.emplace_back(std::make_pair(mod, output_mods.get<cepgen::ParametersList>(mod)));
}

CepGenEventGenerator::~CepGenEventGenerator() { edm::LogInfo("CepGenEventGenerator") << "Destructor called."; }

bool CepGenEventGenerator::initializeForInternalPartons() {
  gen_ = new cepgen::Generator(true /* "safe" mode: start without plugins */);

  auto pproc = proc_params_;
  {  // little treatment to allow for standard CepGen configurations to be copy-pasted in place
    pproc += proc_params_.get<cepgen::ParametersList>("processParameters");
    pproc.erase("processParameters");
    auto& pkin = pproc.operator[]<cepgen::ParametersList>("kinematics");
    pkin += pproc.get<cepgen::ParametersList>("inKinematics");
    pproc.erase("inKinematics");
    pkin += pproc.get<cepgen::ParametersList>("outKinematics");
    pproc.erase("outKinematics");
    if (pproc.has<unsigned long long>("mode"))
      pkin.set<int>("mode", pproc.get<unsigned long long>("mode"));
  }

  gen_->runParameters().setProcess(cepgen::ProcessFactory::get().build(pproc));
  if (!gen_->runParameters().hasProcess())
    throw cms::Exception("CepGenEventGenerator") << "Failed to retrieve a process from the configuration";
  for (const auto& mod : modif_modules_) {
    auto modifier = cepgen::EventModifierFactory::get().build(mod.first, mod.second);
    for (const auto& cfg : mod.second.get<std::vector<std::string> >("preConfiguration"))
      modifier->readString(cfg);
    for (const auto& cfg : mod.second.get<std::vector<std::string> >("processConfiguration"))
      modifier->readString(cfg);
    gen_->runParameters().addModifier(std::move(modifier));
  }
  for (const auto& mod : output_modules_)
    gen_->runParameters().addEventExporter(cepgen::EventExporterFactory::get().build(mod.first, mod.second));

  edm::LogInfo("CepGenEventGenerator") << "Run parameters:\n" << gen_->runParameters();
  const auto xsec = gen_->computeXsection();
  xsec_.set_cross_section(xsec, xsec.uncertainty());
  runInfo().setInternalXSec(GenRunInfoProduct::XSec(xsec, xsec.uncertainty()));
  return true;
}

bool CepGenEventGenerator::generatePartonsAndHadronize() {
  event().reset(new HepMC::CepGenEvent(gen_->next()));
  event()->set_cross_section(xsec_);
  event()->weights().push_back(1.);
  return true;
}

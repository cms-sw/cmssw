// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// Writes truth::GraphMetadata once per run. The configuration of the graph producers and
// of the generator comes from the parameter sets of this job and of the jobs before it,
// so the metadata describes the modules that ran and cannot drift from them.

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/GraphMetadata.h"

namespace {

  // A release name without the quotes that edm::getReleaseVersion() puts around it.
  std::string releaseName(std::string release) {
    if (release.size() >= 2 && release.front() == '"' && release.back() == '"')
      release = release.substr(1, release.size() - 2);
    return release;
  }

  // A module's parameter set and the release of the job that ran it.
  struct ModuleConfig {
    edm::ParameterSet pset;
    std::string release;
  };

  std::optional<ModuleConfig> moduleIn(edm::ParameterSet const& process,
                                       std::string const& label,
                                       std::string const& release) {
    if (label.empty() || !process.existsAs<edm::ParameterSet>(label))
      return std::nullopt;
    return ModuleConfig{process.getParameterSet(label), releaseName(release)};
  }

  // The module with this label in this job, else in the latest earlier job that has it.
  std::optional<ModuleConfig> findModule(edm::ParameterSet const& thisProcess,
                                         edm::ProcessHistory const& history,
                                         std::string const& label) {
    if (auto found = moduleIn(thisProcess, label, edm::getReleaseVersion()))
      return found;
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
      if (auto const* process = edm::pset::Registry::instance()->getMapped(it->parameterSetID()))
        if (auto found = moduleIn(*process, label, it->releaseVersion()))
          return found;
    }
    return std::nullopt;
  }

  std::string moduleType(edm::ParameterSet const& pset) { return pset.getParameter<std::string>("@module_type"); }

  std::string lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
    return text;
  }

  // Every InputTag list of a module whose name ends with one of the suffixes, ignoring case,
  // as encoded tags: simHitCollections and trackerSimHitCollections both end with
  // "simhitcollections".
  std::vector<std::string> collections(edm::ParameterSet const& pset, std::vector<std::string> const& suffixes) {
    std::vector<std::string> out;
    for (auto const& name : pset.getParameterNamesForType<std::vector<edm::InputTag>>()) {
      auto const lowerName = lowercase(name);
      for (auto const& suffix : suffixes) {
        auto const lowerSuffix = lowercase(suffix);
        if (lowerName.ends_with(lowerSuffix)) {
          for (auto const& tag : pset.getParameter<std::vector<edm::InputTag>>(name))
            out.push_back(tag.encode());
          break;
        }
      }
    }
    return out;
  }

  void addEnergyConvention(truth::GraphMetadata& out, std::string subdetector, std::string convention) {
    out.energySubdetectors.push_back(std::move(subdetector));
    out.energyConventions.push_back(std::move(convention));
  }

}  // namespace

class TruthGraphMetadataProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
public:
  explicit TruthGraphMetadataProducer(edm::ParameterSet const& cfg)
      : graphProducer_(cfg.getParameter<std::string>("graphProducer")),
        rawGraphProducer_(cfg.getParameter<std::string>("rawGraphProducer")),
        hitIndexProducer_(cfg.getParameter<std::string>("hitIndexProducer")),
        generatorLabel_(cfg.getParameter<std::string>("generatorLabel")),
        generatorName_(cfg.getParameter<std::string>("generatorName")),
        generatorVersion_(cfg.getParameter<std::string>("generatorVersion")),
        generatorFragment_(cfg.getParameter<std::string>("generatorFragment")) {
    produces<truth::GraphMetadata, edm::Transition::BeginRun>();
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("graphProducer", "truthLogicalGraphProducer")
        ->setComment("Label of the TruthLogicalGraphProducer.");
    desc.add<std::string>("rawGraphProducer", "mix")
        ->setComment("Label of the module that builds the raw graph: a TruthGraphProducer, or the MixingModule.");
    desc.add<std::string>("hitIndexProducer", "truthLogicalGraphHitIndexProducer")
        ->setComment("Label of the TruthLogicalGraphHitIndexProducer. Empty when the job builds no hit index.");
    desc.add<std::string>("generatorLabel", "generator")->setComment("Label of the generator module.");
    desc.add<std::string>("generatorName", "")->setComment("Name of the generator program.");
    desc.add<std::string>("generatorVersion", "")->setComment("Version of the generator program.");
    desc.add<std::string>("generatorFragment", "")->setComment("Name of the generator fragment.");
    descriptions.addWithDefaultLabel(desc);
  }

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}

  void globalBeginRunProduce(edm::Run& run, edm::EventSetup const&) const override {
    auto out = std::make_unique<truth::GraphMetadata>();
    edm::ParameterSet const& thisProcess = edm::getProcessParameterSetContainingModule(moduleDescription());
    edm::ProcessHistory const& history = run.processHistory();
    out->cmsswVersion = releaseName(edm::getReleaseVersion());

    out->generatorName = generatorName_;
    out->generatorVersion = generatorVersion_;
    out->generatorFragment = generatorFragment_;
    if (auto generator = findModule(thisProcess, history, generatorLabel_)) {
      out->generatorModule = moduleType(generator->pset);
      out->generatorCmsswVersion = generator->release;
      out->generatorConfig = generator->pset.dump();
    }

    if (auto raw = findModule(thisProcess, history, rawGraphProducer_)) {
      edm::ParameterSet const& pset = raw->pset;
      if (moduleType(pset) == "MixingModule") {
        auto const& digitizers = pset.getParameterSet("digitizers");
        if (digitizers.existsAs<edm::ParameterSet>("truthGraph")) {
          auto const& accumulator = digitizers.getParameterSet("truthGraph");
          out->buildMode = truth::GraphMetadata::BuildMode::GenSimMixed;
          out->pileupBunchCrossings = accumulator.getParameter<std::vector<int32_t>>("pileupBunchCrossings");
          out->rawGraphProducerConfig = accumulator.dump();
        }
      } else {
        out->buildMode = pset.getParameter<edm::InputTag>("simTracks").label().empty()
                             ? truth::GraphMetadata::BuildMode::GenOnly
                             : truth::GraphMetadata::BuildMode::GenSim;
        out->rawGraphProducerConfig = pset.dump();
      }
    }

    if (auto graph = findModule(thisProcess, history, graphProducer_)) {
      edm::ParameterSet const& pset = graph->pset;
      auto const& selection = pset.getParameterSet("postProcessing");
      out->presetName = selection.getParameter<std::string>("presetName");
      out->signalSeedPdgIds = selection.getParameter<std::vector<int32_t>>("seedPdgIds");
      out->seedHadronFlavors = selection.getParameter<std::vector<int32_t>>("seedHadronFlavors");
      out->reconstructablePdgIds = selection.getParameter<std::vector<int32_t>>("reconstructablePdgIds");
      out->collapseIntermediateGenParticles = selection.getParameter<bool>("collapseIntermediateGenParticles");
      out->dropHitlessSimSubgraphs = selection.getParameter<bool>("dropHitlessSimSubgraphs");
      if (out->dropHitlessSimSubgraphs)
        out->pruningSimHitCollections = collections(pset, {"SimHitCollections"});
      out->graphProducerConfig = pset.dump();
    }

    for (auto const& row : truth::kLevelTable) {
      out->levelNames.emplace_back(row.name);
      out->levelFlags.push_back(static_cast<uint32_t>(row.flag));
    }
    out->levelNames.emplace_back(truth::kSignalLevelName);
    out->levelFlags.push_back(static_cast<uint32_t>(truth::LevelFlag::Signal));

    if (auto index = findModule(thisProcess, history, hitIndexProducer_)) {
      edm::ParameterSet const& pset = index->pset;
      out->hitIndexChannels = pset.getParameter<std::vector<std::string>>("subdetectors");
      out->hitIndexInputs = collections(pset, {"SimHitCollections", "DigiSimLinks"});
      out->hitIndexConfig = pset.dump();
      for (auto const& channel : out->hitIndexChannels) {
        if (channel == "Calo") {
          addEnergyConvention(*out, "HGCAL", "GeV deposited in the active material");
          addEnergyConvention(*out, "ECAL barrel", "GeV deposited in the active material");
          addEnergyConvention(*out, "HCAL barrel, endcap, outer", "GeV deposited in the active material");
          addEnergyConvention(*out,
                              "HF",
                              pset.existsAs<std::vector<double>>("hfSamplingFactors")
                                  ? "GeV: photoelectrons divided by the HF digitizer samplingFactor"
                                  : "photoelectrons (HCalSD stores 1 GeV for each photoelectron)");
        } else {
          addEnergyConvention(*out, channel, "GeV energy loss of the sim hit");
        }
      }
    }

    run.put(std::move(out));
  }

private:
  const std::string graphProducer_;
  const std::string rawGraphProducer_;
  const std::string hitIndexProducer_;
  const std::string generatorLabel_;
  const std::string generatorName_;
  const std::string generatorVersion_;
  const std::string generatorFragment_;
};

DEFINE_FWK_MODULE(TruthGraphMetadataProducer);

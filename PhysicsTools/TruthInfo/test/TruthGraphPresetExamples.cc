// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// One analysis per selection preset, written against the truth graph interface. Each
// function answers the physics question its preset seeds on and prints one line per
// object of interest. The python twin is presetExamples.py, same names, same questions.
//
//   cmsRun presetExamples_cfg.py --preset top step3.root

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/TruthInfo/interface/Branch.h"
#include "PhysicsTools/TruthInfo/interface/TruthLevels.h"
#include "SimDataFormats/TruthInfo/interface/Graph.h"
#include "SimDataFormats/TruthInfo/interface/Particle.h"

namespace {

  using P4 = math::XYZTLorentzVectorD;

  bool isLepton(int32_t pdgId) {
    const int32_t a = std::abs(pdgId);
    return a == 11 || a == 13 || a == 15;
  }

  bool isWeakBoson(int32_t pdgId) {
    const int32_t a = std::abs(pdgId);
    return a == 23 || a == 24;
  }

  // The particles a level names, as views. branchesAtLevel gives the same set as
  // branches; the views are enough where only the particle itself is asked about.
  std::vector<truth::Particle> membersOf(truth::Graph const& graph, truth::Level level) {
    std::vector<truth::Particle> members;
    for (const uint32_t id : truth::levelAntichain(graph, level)) {
      members.emplace_back(&graph, id);
    }
    return members;
  }

  std::vector<truth::Particle> signalRoots(truth::Graph const& graph) {
    std::vector<truth::Particle> roots;
    for (uint32_t id = 0; id < graph.nParticles(); ++id) {
      if (graph.particles()[id].isAtLevel(truth::LevelFlag::Signal)) {
        roots.emplace_back(&graph, id);
      }
    }
    return roots;
  }

  // The copy of a radiating chain whose children are the decay products.
  truth::Particle lastCopy(truth::Particle const& particle) {
    return truth::Particle(particle.graph(), truth::lastCopyOf(*particle.graph(), particle.id()));
  }

  // The first child whose species is in the list, without building the child vector.
  std::optional<truth::Particle> firstChildWithPdgId(truth::Particle const& copy, std::vector<int32_t> const& pdgIds) {
    const truth::Particle particle = lastCopy(copy);
    std::optional<truth::Particle> found;
    particle.forEachChildId([&](uint32_t child) {
      if (found) {
        return;
      }
      const int32_t pdgId = std::abs(particle.graph()->particles()[child].pdgId);
      if (std::find(pdgIds.begin(), pdgIds.end(), pdgId) != pdgIds.end()) {
        found.emplace(particle.graph(), child);
      }
    });
    return found;
  }

  // leptonic, hadronic or none, from the children of a W or a Z. A tau counts as a lepton
  // here, so W -> tau nu is leptonic whatever the tau does next.
  std::string decayMode(truth::Particle const& copy) {
    const truth::Particle boson = lastCopy(copy);
    std::string mode = "none";
    boson.forEachChildId([&](uint32_t child) {
      const int32_t pdgId = std::abs(boson.graph()->particles()[child].pdgId);
      if (isLepton(pdgId)) {
        mode = "leptonic";
      } else if (pdgId <= 6 && mode == "none") {
        mode = "hadronic";
      }
    });
    return mode;
  }

  // The other particles produced where this one was.
  std::vector<truth::Particle> productionSiblings(truth::Particle const& particle) {
    std::vector<truth::Particle> siblings;
    for (auto const& vertex : particle.productionVertices()) {
      for (auto const& out : vertex.outgoingParticles()) {
        if (out != particle) {
          siblings.push_back(out);
        }
      }
    }
    return siblings;
  }

  std::string line(truth::Particle const& p) {
    return std::to_string(p.pdgId()) + " pt " + std::to_string(p.momentum().pt()) + " GeV";
  }

  // --- gun: each gun particle is its own signal -----------------------------------------
  void gun(truth::Graph const& graph) {
    for (auto const& seed : signalRoots(graph)) {
      const truth::Branch branch(&graph, seed.id());
      std::size_t products = 0;
      std::size_t atCalo = 0;
      for (auto const& member : branch.members()) {
        products += member.data().isAtLevel(truth::LevelFlag::ReconstructableFromSignal) ? 1 : 0;
        atCalo += member.data().isAtLevel(truth::LevelFlag::CaloBoundary) ? 1 : 0;
      }
      edm::LogPrint("presetExample") << "gun " << seed.pdgId() << " E " << seed.momentum().energy()
                                     << " GeV: " << products << " reconstructable products, " << atCalo
                                     << " descendants reach the calorimeter";
    }
  }

  // --- resonance: the boson and its leptonic legs ---------------------------------------
  void resonance(truth::Graph const& graph) {
    for (auto const& z : signalRoots(graph)) {
      std::vector<truth::Particle> legs;
      for (auto const& child : lastCopy(z).children()) {
        if (isLepton(child.pdgId())) {
          legs.push_back(child);
        }
      }
      if (legs.size() != 2) {
        edm::LogPrint("presetExample") << "resonance " << z.pdgId() << ": decay mode " << decayMode(z);
        continue;
      }
      const P4 dilepton = legs[0].momentum() + legs[1].momentum();
      edm::LogPrint("presetExample") << "resonance " << z.pdgId() << " -> " << legs[0].pdgId() << " " << legs[1].pdgId()
                                     << ": m(ll) " << dilepton.mass() << " GeV, generator mass " << z.momentum().mass()
                                     << " GeV";
    }
  }

  // --- vbf: the Higgs and the two tagging quarks -----------------------------------------
  void vbf(truth::Graph const& graph) {
    const auto higgs = signalRoots(graph);
    std::vector<truth::Particle> tagging;
    for (auto const& parton : membersOf(graph, truth::Level::PartonJets)) {
      if (parton.data().isSignal()) {
        tagging.push_back(parton);
      }
    }
    if (tagging.size() < 2) {
      edm::LogPrint("presetExample") << "vbf: " << higgs.size() << " Higgs, " << tagging.size() << " tagging partons";
      return;
    }
    std::sort(tagging.begin(), tagging.end(), [](auto const& a, auto const& b) {
      return a.momentum().pt() > b.momentum().pt();
    });
    const P4 dijet = tagging[0].momentum() + tagging[1].momentum();
    edm::LogPrint("presetExample") << "vbf: " << higgs.size() << " Higgs, tagging partons " << tagging[0].pdgId() << " "
                                   << tagging[1].pdgId() << ": m(jj) " << dijet.mass() << " GeV, |delta eta| "
                                   << std::abs(tagging[0].momentum().eta() - tagging[1].momentum().eta());
  }

  // --- ggf: the Higgs and what the detector can see of it ---------------------------------
  void ggf(truth::Graph const& graph) {
    for (auto const& higgs : signalRoots(graph)) {
      const truth::Branch branch(&graph, higgs.id());
      std::size_t products = 0;
      double visible = 0.;
      for (auto const& member : branch.members()) {
        if (!member.data().isAtLevel(truth::LevelFlag::ReconstructableFromSignal)) {
          continue;
        }
        ++products;
        visible += truth::isInvisible(member.pdgId()) ? 0. : member.momentum().energy();
      }
      edm::LogPrint("presetExample") << "ggf: Higgs E " << higgs.momentum().energy() << " GeV -> " << products
                                     << " reconstructable products, visible fraction "
                                     << visible / higgs.momentum().energy();
    }
  }

  // --- vh: the Higgs and the boson produced with it -----------------------------------------
  void vh(truth::Graph const& graph) {
    for (auto const& higgs : signalRoots(graph)) {
      for (auto const& sibling : productionSiblings(higgs)) {
        if (isWeakBoson(sibling.pdgId())) {
          edm::LogPrint("presetExample") << "vh: Higgs with " << line(sibling) << ", boson decay "
                                         << decayMode(sibling);
        }
      }
    }
  }

  // --- top: the two tops, their b and W, and the event class -------------------------------
  void top(truth::Graph const& graph) {
    int leptonic = 0;
    for (auto const& t : signalRoots(graph)) {
      const auto b = firstChildWithPdgId(t, {5});
      const auto w = firstChildWithPdgId(t, {24});
      const std::string mode = w ? decayMode(*w) : "none";
      leptonic += (mode == "leptonic") ? 1 : 0;
      edm::LogPrint("presetExample") << "top " << t.pdgId() << ": b " << (b ? "yes" : "no") << ", W " << mode << ", "
                                     << t.descendants().size() << " descendants";
    }
    const char* eventClass = leptonic == 0 ? "all hadronic" : leptonic == 1 ? "semileptonic" : "dilepton";
    edm::LogPrint("presetExample") << "top event class: " << eventClass;
  }

  // --- singletop: the top and its production partner ---------------------------------------
  void singletop(truth::Graph const& graph) {
    for (auto const& t : signalRoots(graph)) {
      for (auto const& partner : productionSiblings(t)) {
        edm::LogPrint("presetExample") << "singletop: top with partner " << line(partner);
      }
    }
  }

  // --- diboson: the bosons, their modes and their mass -------------------------------------
  void diboson(truth::Graph const& graph) {
    std::vector<truth::Particle> bosons;
    for (auto const& root : signalRoots(graph)) {
      if (isWeakBoson(root.pdgId())) {
        bosons.push_back(root);
      }
    }
    if (bosons.size() >= 2) {
      edm::LogPrint("presetExample") << "diboson: m(VV) " << (bosons[0].momentum() + bosons[1].momentum()).mass()
                                     << " GeV";
    }
    for (auto const& boson : bosons) {
      edm::LogPrint("presetExample") << "  " << line(boson) << ", decay " << decayMode(boson);
    }
  }

  // --- heavyflavor: the b hadrons and their flight ----------------------------------------
  void heavyflavor(truth::Graph const& graph) {
    for (auto const& branch : truth::branchesAtLevel(graph, truth::Level::BHadrons)) {
      const truth::Particle hadron = branch.root();
      const auto production = hadron.productionVertices();
      const auto decay = hadron.decayVertices();
      std::string flight = "n/a";
      if (!production.empty() && !decay.empty()) {
        const auto& x0 = production.front().data().position;
        const auto& x1 = decay.front().data().position;
        flight =
            std::to_string(std::sqrt((x1.x() - x0.x()) * (x1.x() - x0.x()) + (x1.y() - x0.y()) * (x1.y() - x0.y()) +
                                     (x1.z() - x0.z()) * (x1.z() - x0.z()))) +
            " cm";
      }
      std::size_t charm = 0;
      for (auto const& member : branch.members()) {
        charm += member.data().isAtLevel(truth::LevelFlag::CHadrons) ? 1 : 0;
      }
      edm::LogPrint("presetExample") << "heavyflavor: " << line(hadron) << ", flight " << flight << ", " << charm
                                     << " charm hadron" << (charm == 1 ? "" : "s") << " below";
    }
  }

  // --- full: the whole event, signal and pile-up apart --------------------------------------
  void full(truth::Graph const& graph) {
    std::map<uint64_t, std::size_t> particlesOf;
    for (auto const& particle : graph.particles()) {
      ++particlesOf[particle.eventId];
    }
    for (auto const& [eventId, count] : particlesOf) {
      edm::LogPrint("presetExample") << "full: interaction bx " << truth::bunchCrossingOf(eventId) << " index "
                                     << truth::eventIndexOf(eventId) << ": " << count << " particles";
    }
    for (const bool signal : {true, false}) {
      std::size_t objects = 0;
      std::size_t withoutMomentum = 0;
      double energy = 0.;
      for (auto const& member : membersOf(graph, truth::Level::ReconstructableFinalState)) {
        if (member.data().isSignal() != signal) {
          continue;
        }
        ++objects;
        energy += member.momentum().energy();
        withoutMomentum += member.data().hasMomentum() ? 0 : 1;
      }
      edm::LogPrint("presetExample") << "full: " << (signal ? "signal" : "pileup") << " reconstructable final state "
                                     << objects << " objects, " << energy << " GeV, " << withoutMomentum
                                     << " without momentum";
    }
  }

  const std::map<std::string, std::function<void(truth::Graph const&)>> kPresets = {
      {"gun", gun},
      {"resonance", resonance},
      {"vbf", vbf},
      {"ggf", ggf},
      {"vh", vh},
      {"top", top},
      {"singletop", singletop},
      {"diboson", diboson},
      {"heavyflavor", heavyflavor},
      {"full", full},
  };

}  // namespace

class TruthGraphPresetExamples : public edm::global::EDAnalyzer<> {
public:
  explicit TruthGraphPresetExamples(edm::ParameterSet const& pset)
      : token_(consumes<truth::Graph>(pset.getParameter<edm::InputTag>("src"))) {
    for (auto const& name : pset.getParameter<std::vector<std::string>>("presets")) {
      const auto it = kPresets.find(name);
      if (it == kPresets.end()) {
        throw cms::Exception("Configuration") << "unknown preset '" << name << "'";
      }
      examples_.push_back(it->second);
      names_.push_back(name);
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("truthLogicalGraphProducer"));
    std::vector<std::string> all;
    for (auto const& [name, example] : kPresets) {
      all.push_back(name);
    }
    desc.add<std::vector<std::string>>("presets", all)->setComment("The examples to run, by preset name");
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
    auto const& graph = event.get(token_);
    edm::LogPrint("presetExample") << "== event " << event.id().event();
    for (std::size_t i = 0; i < examples_.size(); ++i) {
      edm::LogPrint("presetExample") << "-- " << names_[i];
      examples_[i](graph);
    }
  }

private:
  const edm::EDGetTokenT<truth::Graph> token_;
  std::vector<std::function<void(truth::Graph const&)>> examples_;
  std::vector<std::string> names_;
};

DEFINE_FWK_MODULE(TruthGraphPresetExamples);

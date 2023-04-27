
#include <vector>
#include <numeric>
#include <string>

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "DataFormats/Math/interface/deltaR.h"

// bitwise emulation headers
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"
#include "DataFormats/L1TParticleFlow/interface/gt_datatypes.h"

class L1SeedConePFJetProducer : public edm::global::EDProducer<> {
public:
  explicit L1SeedConePFJetProducer(const edm::ParameterSet&);
  ~L1SeedConePFJetProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& description);

private:
  /// ///////////////// ///
  /// MANDATORY METHODS ///
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  /// ///////////////// ///

  const float coneSize;
  const unsigned nJets;
  const bool HW;
  const bool debug;
  const bool doCorrections;
  L1SCJetEmu emulator;
  edm::EDGetTokenT<std::vector<l1t::PFCandidate>> l1PFToken;
  l1tpf::corrector corrector;

  std::vector<l1t::PFJet> processEvent_SW(std::vector<edm::Ptr<l1t::PFCandidate>>& parts) const;
  std::vector<l1t::PFJet> processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& parts) const;

  l1t::PFJet makeJet_SW(const std::vector<edm::Ptr<l1t::PFCandidate>>& parts) const;

  std::pair<std::vector<L1SCJetEmu::Particle>, std::unordered_map<const l1t::PFCandidate*, edm::Ptr<l1t::PFCandidate>>>
  convertEDMToHW(std::vector<edm::Ptr<l1t::PFCandidate>>& edmParticles) const;

  std::vector<l1t::PFJet> convertHWToEDM(
      std::vector<L1SCJetEmu::Jet> hwJets,
      std::unordered_map<const l1t::PFCandidate*, edm::Ptr<l1t::PFCandidate>> constituentMap) const;
};

L1SeedConePFJetProducer::L1SeedConePFJetProducer(const edm::ParameterSet& cfg)
    : coneSize(cfg.getParameter<double>("coneSize")),
      nJets(cfg.getParameter<unsigned>("nJets")),
      HW(cfg.getParameter<bool>("HW")),
      debug(cfg.getParameter<bool>("debug")),
      doCorrections(cfg.getParameter<bool>("doCorrections")),
      emulator(L1SCJetEmu(debug, coneSize, nJets)),
      l1PFToken(consumes<std::vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))) {
  produces<l1t::PFJetCollection>();
  if (doCorrections) {
    corrector = l1tpf::corrector(
        cfg.getParameter<std::string>("correctorFile"), cfg.getParameter<std::string>("correctorDir"), -1., debug, HW);
  }
}

void L1SeedConePFJetProducer::produce(edm::StreamID /*unused*/,
                                      edm::Event& iEvent,
                                      const edm::EventSetup& iSetup) const {
  std::unique_ptr<l1t::PFJetCollection> newPFJetCollection(new l1t::PFJetCollection);

  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(l1PFToken, l1PFCandidates);

  std::vector<edm::Ptr<l1t::PFCandidate>> particles;
  for (unsigned i = 0; i < (*l1PFCandidates).size(); i++) {
    particles.push_back(edm::Ptr<l1t::PFCandidate>(l1PFCandidates, i));
  }

  std::vector<l1t::PFJet> jets;
  if (HW) {
    jets = processEvent_HW(particles);
  } else {
    jets = processEvent_SW(particles);
  }

  std::sort(jets.begin(), jets.end(), [](l1t::PFJet i, l1t::PFJet j) { return (i.pt() > j.pt()); });
  newPFJetCollection->swap(jets);
  iEvent.put(std::move(newPFJetCollection));
}

/////////////
// DESTRUCTOR
L1SeedConePFJetProducer::~L1SeedConePFJetProducer() {}

l1t::PFJet L1SeedConePFJetProducer::makeJet_SW(const std::vector<edm::Ptr<l1t::PFCandidate>>& parts) const {
  l1t::PFCandidate seed = *parts.at(0);

  auto sumpt = [](float a, const edm::Ptr<l1t::PFCandidate>& b) { return a + b->pt(); };

  // Sum the pt
  float pt = std::accumulate(parts.begin(), parts.end(), 0., sumpt);

  // pt weighted d eta
  std::vector<float> pt_deta;
  pt_deta.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pt_deta.begin(), [&seed, &pt](const edm::Ptr<l1t::PFCandidate>& part) {
    return (part->pt() / pt) * (part->eta() - seed.eta());
  });
  // Accumulate the pt weighted etas. Init to the seed eta, start accumulating at begin()+1 to skip seed
  float eta = std::accumulate(pt_deta.begin() + 1, pt_deta.end(), seed.eta());

  // pt weighted d phi
  std::vector<float> pt_dphi;
  pt_dphi.resize(parts.size());
  std::transform(parts.begin(), parts.end(), pt_dphi.begin(), [&seed, &pt](const edm::Ptr<l1t::PFCandidate>& part) {
    return (part->pt() / pt) * reco::deltaPhi(part->phi(), seed.phi());
  });
  // Accumulate the pt weighted phis. Init to the seed phi, start accumulating at begin()+1 to skip seed
  float phi = std::accumulate(pt_dphi.begin() + 1, pt_dphi.end(), seed.phi());

  l1t::PFJet jet(pt, eta, phi);
  for (auto it = parts.begin(); it != parts.end(); it++) {
    jet.addConstituent(*it);
  }

  if (doCorrections) {
    jet.calibratePt(corrector.correctedPt(jet.pt(), jet.eta()));
  }

  return jet;
}

std::vector<l1t::PFJet> L1SeedConePFJetProducer::processEvent_SW(std::vector<edm::Ptr<l1t::PFCandidate>>& work) const {
  // The floating point algorithm simulation
  std::stable_sort(work.begin(), work.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });
  std::vector<l1t::PFJet> jets;
  jets.reserve(nJets);
  while (!work.empty() && jets.size() < nJets) {
    // Take the first (highest pt) candidate as a seed
    edm::Ptr<l1t::PFCandidate> seed = work.at(0);
    // Get the particles within a coneSize of the seed
    std::vector<edm::Ptr<l1t::PFCandidate>> particlesInCone;
    std::copy_if(
        work.begin(), work.end(), std::back_inserter(particlesInCone), [&](const edm::Ptr<l1t::PFCandidate>& part) {
          return reco::deltaR<l1t::PFCandidate, l1t::PFCandidate>(*seed, *part) <= coneSize;
        });
    jets.push_back(makeJet_SW(particlesInCone));
    // remove the clustered particles
    work.erase(std::remove_if(work.begin(),
                              work.end(),
                              [&](const edm::Ptr<l1t::PFCandidate>& part) {
                                return reco::deltaR<l1t::PFCandidate, l1t::PFCandidate>(*seed, *part) <= coneSize;
                              }),
               work.end());
  }
  return jets;
}

std::vector<l1t::PFJet> L1SeedConePFJetProducer::processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& work) const {
  // The fixed point emulator
  // Convert the EDM format to the hardware format, and call the standalone emulator
  std::pair<std::vector<L1SCJetEmu::Particle>, std::unordered_map<const l1t::PFCandidate*, edm::Ptr<l1t::PFCandidate>>>
      particles = convertEDMToHW(work);
  std::vector<L1SCJetEmu::Jet> jets = emulator.emulateEvent(particles.first);
  return convertHWToEDM(jets, particles.second);
}

std::pair<std::vector<L1SCJetEmu::Particle>, std::unordered_map<const l1t::PFCandidate*, edm::Ptr<l1t::PFCandidate>>>
L1SeedConePFJetProducer::convertEDMToHW(std::vector<edm::Ptr<l1t::PFCandidate>>& edmParticles) const {
  std::vector<l1ct::PuppiObjEmu> hwParticles;
  std::unordered_map<const l1t::PFCandidate*, edm::Ptr<l1t::PFCandidate>> candidateMap;
  std::for_each(edmParticles.begin(), edmParticles.end(), [&](edm::Ptr<l1t::PFCandidate>& edmParticle) {
    l1ct::PuppiObjEmu particle;
    particle.initFromBits(edmParticle->encodedPuppi64());
    particle.srcCand = edmParticle.get();
    candidateMap.insert(std::make_pair(edmParticle.get(), edmParticle));
    hwParticles.push_back(particle);
  });
  return std::make_pair(hwParticles, candidateMap);
}

std::vector<l1t::PFJet> L1SeedConePFJetProducer::convertHWToEDM(
    std::vector<L1SCJetEmu::Jet> hwJets,
    std::unordered_map<const l1t::PFCandidate*, edm::Ptr<l1t::PFCandidate>> constituentMap) const {
  std::vector<l1t::PFJet> edmJets;
  std::for_each(hwJets.begin(), hwJets.end(), [&](L1SCJetEmu::Jet jet) {
    if (doCorrections) {
      float correctedPt = corrector.correctedPt(jet.floatPt(), jet.floatEta());
      jet.hwPt = correctedPt;
    }
    l1gt::Jet gtJet = jet.toGT();
    l1t::PFJet edmJet(l1gt::Scales::floatPt(gtJet.v3.pt),
                      l1gt::Scales::floatEta(gtJet.v3.eta),
                      l1gt::Scales::floatPhi(gtJet.v3.phi),
                      /*mass=*/0.,
                      gtJet.v3.pt.V,
                      gtJet.v3.eta.V,
                      gtJet.v3.phi.V);
    edmJet.setEncodedJet(jet.toGT().pack());
    // get back the references to the constituents
    std::vector<edm::Ptr<l1t::PFCandidate>> constituents;
    std::for_each(jet.constituents.begin(), jet.constituents.end(), [&](auto constituent) {
      edmJet.addConstituent(constituentMap[constituent.srcCand]);
    });
    edmJets.push_back(edmJet);
  });
  return edmJets;
}

void L1SeedConePFJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1PFObjects", edm::InputTag("l1tLayer1", "Puppi"));
  desc.add<uint32_t>("nJets", 16);
  desc.add<double>("coneSize", 0.4);
  desc.add<bool>("HW", false);
  desc.add<bool>("debug", false);
  desc.add<bool>("doCorrections", false);
  desc.add<std::string>("correctorFile", "");
  desc.add<std::string>("correctorDir", "");
}

DEFINE_FWK_MODULE(L1SeedConePFJetProducer);

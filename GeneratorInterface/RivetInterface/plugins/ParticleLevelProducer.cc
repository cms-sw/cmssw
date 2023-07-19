#include <memory>

#include "GeneratorInterface/RivetInterface/interface/ParticleLevelProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"
#include "CommonTools/Utils/interface/PtComparator.h"

#include "Rivet/Analysis.hh"

using namespace std;
using namespace edm;
using namespace reco;
using namespace Rivet;

ParticleLevelProducer::ParticleLevelProducer(const edm::ParameterSet& pset)
    : srcToken_(consumes<edm::HepMCProduct>(pset.getParameter<edm::InputTag>("src"))), pset_(pset) {
  usesResource("Rivet");
  genVertex_ = reco::Particle::Point(0, 0, 0);

  produces<reco::GenParticleCollection>("neutrinos");
  produces<reco::GenParticleCollection>("photons");
  produces<reco::GenJetCollection>("leptons");
  produces<reco::GenJetCollection>("jets");
  produces<reco::GenJetCollection>("fatjets");
  produces<reco::GenParticleCollection>("consts");
  produces<reco::GenParticleCollection>("tags");
  produces<reco::METCollection>("mets");
}

void ParticleLevelProducer::addGenJet(Rivet::Jet jet,
                                      std::unique_ptr<reco::GenJetCollection>& jets,
                                      std::unique_ptr<reco::GenParticleCollection>& consts,
                                      edm::RefProd<reco::GenParticleCollection>& constsRefHandle,
                                      int& iConstituent,
                                      std::unique_ptr<reco::GenParticleCollection>& tags,
                                      edm::RefProd<reco::GenParticleCollection>& tagsRefHandle,
                                      int& iTag) {
  const auto pjet = jet.pseudojet();

  reco::GenJet genJet;
  genJet.setP4(p4(jet));
  genJet.setVertex(genVertex_);
  if (jet.bTagged())
    genJet.setPdgId(5);
  else if (jet.cTagged())
    genJet.setPdgId(4);
  genJet.setJetArea(pjet.has_area() ? pjet.area() : 0);

  for (auto const& p : jet.particles()) {
    auto pp4 = p4(p);
    bool match = false;
    int iMatch = -1;
    for (auto const& q : *consts) {
      ++iMatch;
      if (q.p4() == pp4) {
        match = true;
        break;
      }
    }
    if (match) {
      genJet.addDaughter(edm::refToPtr(reco::GenParticleRef(constsRefHandle, iMatch)));
    } else {
      consts->push_back(reco::GenParticle(p.charge(), pp4, genVertex_, p.pid(), 1, true));
      genJet.addDaughter(edm::refToPtr(reco::GenParticleRef(constsRefHandle, ++iConstituent)));
    }
  }
  for (auto const& p : jet.tags()) {
    // The tag particles are accessible as jet daughters, so scale down p4 for safety.
    // p4 needs to be multiplied by 1e20 for fragmentation analysis.
    auto pp4 = p4(p) * 1e-20;
    bool match = false;
    int iMatch = -1;
    for (auto const& q : *tags) {
      ++iMatch;
      if (q.p4() == pp4) {
        match = true;
        break;
      }
    }
    if (match) {
      genJet.addDaughter(edm::refToPtr(reco::GenParticleRef(tagsRefHandle, iMatch)));
    } else {
      tags->push_back(reco::GenParticle(p.charge(), p4(p) * 1e-20, genVertex_, p.pid(), 2, true));
      genJet.addDaughter(edm::refToPtr(reco::GenParticleRef(tagsRefHandle, ++iTag)));
      // Also save lepton+neutrino daughters of tag particles
      int iTagMother = iTag;
      for (auto const& d : p.constituents()) {
        ++iTag;
        int d_status = (d.isStable()) ? 1 : 2;
        tags->push_back(reco::GenParticle(d.charge(), p4(d) * 1e-20, genVertex_, d.pid(), d_status, true));
        tags->at(iTag).addMother(reco::GenParticleRef(tagsRefHandle, iTagMother));
        tags->at(iTagMother).addDaughter(reco::GenParticleRef(tagsRefHandle, iTag));
        genJet.addDaughter(edm::refToPtr(reco::GenParticleRef(tagsRefHandle, iTag)));
      }
    }
  }

  jets->push_back(genJet);
}

void ParticleLevelProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  using namespace Rivet;
  typedef reco::Candidate::LorentzVector LorentzVector;

  std::unique_ptr<reco::GenParticleCollection> neutrinos(new reco::GenParticleCollection);
  std::unique_ptr<reco::GenParticleCollection> photons(new reco::GenParticleCollection);
  std::unique_ptr<reco::GenJetCollection> leptons(new reco::GenJetCollection);
  std::unique_ptr<reco::GenJetCollection> jets(new reco::GenJetCollection);
  std::unique_ptr<reco::GenJetCollection> fatjets(new reco::GenJetCollection);
  std::unique_ptr<reco::GenParticleCollection> consts(new reco::GenParticleCollection);
  std::unique_ptr<reco::GenParticleCollection> tags(new reco::GenParticleCollection);
  std::unique_ptr<reco::METCollection> mets(new reco::METCollection);
  auto constsRefHandle = event.getRefBeforePut<reco::GenParticleCollection>("consts");
  auto tagsRefHandle = event.getRefBeforePut<reco::GenParticleCollection>("tags");

  edm::Handle<HepMCProduct> srcHandle;
  event.getByToken(srcToken_, srcHandle);

  const HepMC::GenEvent* genEvent = srcHandle->GetEvent();

  if (!rivetAnalysis_ || !rivetAnalysis_->hasProjection("FS")) {
    rivetAnalysis_ = new Rivet::RivetAnalysis(pset_);
    analysisHandler_ = std::make_unique<Rivet::AnalysisHandler>();

    analysisHandler_->setIgnoreBeams(true);
    analysisHandler_->addAnalysis(rivetAnalysis_);
  }

  analysisHandler_->analyze(*genEvent);

  // Convert into edm objects
  // Prompt neutrinos
  for (auto const& p : rivetAnalysis_->neutrinos()) {
    neutrinos->push_back(reco::GenParticle(p.charge(), p4(p), genVertex_, p.pid(), 1, true));
  }
  std::sort(neutrinos->begin(), neutrinos->end(), GreaterByPt<reco::Candidate>());

  // Photons
  for (auto const& p : rivetAnalysis_->photons()) {
    photons->push_back(reco::GenParticle(p.charge(), p4(p), genVertex_, p.pid(), 1, true));
  }
  std::sort(photons->begin(), photons->end(), GreaterByPt<reco::Candidate>());

  // Prompt leptons
  int iConstituent = -1;
  int iTag = -1;
  for (auto const& lepton : rivetAnalysis_->leptons()) {
    reco::GenJet lepJet;
    lepJet.setP4(p4(lepton));
    lepJet.setVertex(genVertex_);
    lepJet.setPdgId(lepton.pid());
    lepJet.setCharge(lepton.charge());

    for (auto const& p : lepton.constituents()) {
      // ghost taus (momentum scaled with 10e-20 in RivetAnalysis.h already)
      if (p.abspid() == 15) {
        tags->push_back(reco::GenParticle(p.charge(), p4(p), genVertex_, p.pid(), 2, true));
        lepJet.addDaughter(edm::refToPtr(reco::GenParticleRef(tagsRefHandle, ++iTag)));
      }
      // electrons, muons, photons
      else {
        consts->push_back(reco::GenParticle(p.charge(), p4(p), genVertex_, p.pid(), 1, true));
        lepJet.addDaughter(edm::refToPtr(reco::GenParticleRef(constsRefHandle, ++iConstituent)));
      }
    }

    leptons->push_back(lepJet);
  }
  std::sort(leptons->begin(), leptons->end(), GreaterByPt<reco::GenJet>());

  // Jets with constituents and tag particles
  for (const auto& jet : rivetAnalysis_->jets()) {
    addGenJet(jet, jets, consts, constsRefHandle, iConstituent, tags, tagsRefHandle, iTag);
  }
  for (const auto& jet : rivetAnalysis_->fatjets()) {
    addGenJet(jet, fatjets, consts, constsRefHandle, iConstituent, tags, tagsRefHandle, iTag);
  }

  // MET
  reco::Candidate::LorentzVector metP4(rivetAnalysis_->met().x(),
                                       rivetAnalysis_->met().y(),
                                       0.,
                                       sqrt(pow(rivetAnalysis_->met().x(), 2) + pow(rivetAnalysis_->met().y(), 2)));
  mets->push_back(reco::MET(metP4, genVertex_));

  event.put(std::move(neutrinos), "neutrinos");
  event.put(std::move(photons), "photons");
  event.put(std::move(leptons), "leptons");
  event.put(std::move(jets), "jets");
  event.put(std::move(fatjets), "fatjets");
  event.put(std::move(consts), "consts");
  event.put(std::move(tags), "tags");
  event.put(std::move(mets), "mets");
}

DEFINE_FWK_MODULE(ParticleLevelProducer);

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <tuple>
#include <string>
#include <vector>
#include <TLorentzVector.h>

using namespace std;

class GenPartIsoProducer : public edm::stream::EDProducer<> {
public:
  explicit GenPartIsoProducer(const edm::ParameterSet& iConfig)
      : finalGenParticleToken(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genPart"))),
        packedGenParticlesToken(
            consumes<pat::PackedGenParticleCollection>(iConfig.getParameter<edm::InputTag>("packedGenPart"))),
        additionalPdgId_(iConfig.getParameter<int>("additionalPdgId")) {
    produces<edm::ValueMap<float>>();
  }
  ~GenPartIsoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  float computeIso(TLorentzVector thisPart,
                   edm::Handle<pat::PackedGenParticleCollection> packedGenParticles,
                   std::set<int> gen_fsrset,
                   bool skip_leptons);
  std::vector<float> Lepts_RelIso;
  edm::EDGetTokenT<reco::GenParticleCollection> finalGenParticleToken;
  edm::EDGetTokenT<pat::PackedGenParticleCollection> packedGenParticlesToken;
  int additionalPdgId_;
};

GenPartIsoProducer::~GenPartIsoProducer() {}

void GenPartIsoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  auto finalParticles = iEvent.getHandle(finalGenParticleToken);
  auto packedGenParticles = iEvent.getHandle(packedGenParticlesToken);

  reco::GenParticleCollection::const_iterator genPart;

  Lepts_RelIso.clear();

  for (genPart = finalParticles->begin(); genPart != finalParticles->end(); genPart++) {
    if (abs(genPart->pdgId()) == 11 || abs(genPart->pdgId()) == 13 || abs(genPart->pdgId()) == 15) {
      TLorentzVector lep_dressed;
      lep_dressed.SetPtEtaPhiE(genPart->pt(), genPart->eta(), genPart->phi(), genPart->energy());
      std::set<int> gen_fsrset;
      for (size_t k = 0; k < packedGenParticles->size(); k++) {
        if ((*packedGenParticles)[k].status() != 1)
          continue;
        if ((*packedGenParticles)[k].pdgId() != 22)
          continue;
        double this_dR_lgamma = reco::deltaR(
            genPart->eta(), genPart->phi(), (*packedGenParticles)[k].eta(), (*packedGenParticles)[k].phi());
        bool idmatch = false;
        if ((*packedGenParticles)[k].mother(0)->pdgId() == genPart->pdgId())
          idmatch = true;
        const reco::Candidate* mother = (*packedGenParticles)[k].mother(0);
        for (size_t m = 0; m < mother->numberOfMothers(); m++) {
          if ((*packedGenParticles)[k].mother(m)->pdgId() == genPart->pdgId())
            idmatch = true;
        }
        if (!idmatch)
          continue;
        if (this_dR_lgamma < 0.3) {
          gen_fsrset.insert(k);
          TLorentzVector gamma;
          gamma.SetPtEtaPhiE((*packedGenParticles)[k].pt(),
                             (*packedGenParticles)[k].eta(),
                             (*packedGenParticles)[k].phi(),
                             (*packedGenParticles)[k].energy());
          lep_dressed = lep_dressed + gamma;
        }
      }
      float this_GENiso = 0.0;
      TLorentzVector thisLep;
      thisLep.SetPtEtaPhiM(lep_dressed.Pt(), lep_dressed.Eta(), lep_dressed.Phi(), lep_dressed.M());
      this_GENiso = computeIso(thisLep, packedGenParticles, gen_fsrset, true);
      Lepts_RelIso.push_back(this_GENiso);
    } else if (abs(genPart->pdgId()) == additionalPdgId_) {
      float this_GENiso = 0.0;
      std::set<int> gen_fsrset_nolep;
      TLorentzVector thisPart;
      thisPart.SetPtEtaPhiE(genPart->pt(), genPart->eta(), genPart->phi(), genPart->energy());
      this_GENiso = computeIso(thisPart, packedGenParticles, gen_fsrset_nolep, false);
      Lepts_RelIso.push_back(this_GENiso);
    } else {
      float this_GENiso = 0.0;
      Lepts_RelIso.push_back(this_GENiso);
    }
  }

  auto isoV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerIsoMap(*isoV);
  fillerIsoMap.insert(finalParticles, Lepts_RelIso.begin(), Lepts_RelIso.end());
  fillerIsoMap.fill();
  iEvent.put(std::move(isoV));
}

float GenPartIsoProducer::computeIso(TLorentzVector thisPart,
                                     edm::Handle<pat::PackedGenParticleCollection> packedGenParticles,
                                     std::set<int> gen_fsrset,
                                     bool skip_leptons) {
  double this_GENiso = 0.0;
  for (size_t k = 0; k < packedGenParticles->size(); k++) {
    if ((*packedGenParticles)[k].status() != 1)
      continue;
    if (abs((*packedGenParticles)[k].pdgId()) == 12 || abs((*packedGenParticles)[k].pdgId()) == 14 ||
        abs((*packedGenParticles)[k].pdgId()) == 16)
      continue;
    if (reco::deltaR(thisPart.Eta(), thisPart.Phi(), (*packedGenParticles)[k].eta(), (*packedGenParticles)[k].phi()) <
        0.001)
      continue;
    if (skip_leptons == true) {
      if ((abs((*packedGenParticles)[k].pdgId()) == 11 || abs((*packedGenParticles)[k].pdgId()) == 13))
        continue;
      if (gen_fsrset.find(k) != gen_fsrset.end())
        continue;
    }
    double this_dRvL_nolep =
        reco::deltaR(thisPart.Eta(), thisPart.Phi(), (*packedGenParticles)[k].eta(), (*packedGenParticles)[k].phi());
    if (this_dRvL_nolep < 0.3) {
      this_GENiso = this_GENiso + (*packedGenParticles)[k].pt();
    }
  }
  this_GENiso = this_GENiso / thisPart.Pt();
  return this_GENiso;
}

void GenPartIsoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Description of external inputs requered by this module
  // genPart: collection of gen particles for which to compute Iso
  // packedGenPart: collection of particles to be used for leptons dressing
  // additionalPdgId: additional particle (besides leptons) for which Iso is computed
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genPart")->setComment("input physics object collection");
  desc.add<edm::InputTag>("packedGenPart")->setComment("input stable hadrons collection");
  desc.add<int>("additionalPdgId")->setComment("additional pdgId for Iso computation");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GenPartIsoProducer);

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

int MotherID_geniso(const reco::GenParticle* p){
    int ID = 0;
    int nMo = p->numberOfMothers();
    const reco::Candidate* g = (const reco::Candidate*) p;
    while(nMo>0){
        if(g->pdgId()!=g->mother()->pdgId()) { ID = g->mother()->pdgId(); return ID;  }
        else {
            g = (g->mother());
            nMo = g->numberOfMothers();
        }
    }
    return ID;
}

class GenPartIsoProducer : public edm::stream::EDProducer<> {
public:
  explicit GenPartIsoProducer(const edm::ParameterSet& iConfig):
       finalGenParticleToken(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
       produces<edm::ValueMap<float>>();
       packedgenParticlesToken = consumes<edm::View<pat::PackedGenParticle> > (edm::InputTag("packedGenParticles"));
   }
  ~GenPartIsoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  float computeIso(TLorentzVector thisPart, edm::Handle<edm::View<pat::PackedGenParticle> > packedgenParticles,std::set<int> gen_fsrset, bool skip_leptons);
  std::vector<float> Lepts_RelIso;
  edm::EDGetTokenT<edm::View<pat::PackedGenParticle> > packedgenParticlesToken;
  edm::EDGetTokenT<reco::GenParticleCollection> finalGenParticleToken;
};

GenPartIsoProducer::~GenPartIsoProducer() {}

void GenPartIsoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<edm::View<pat::PackedGenParticle> > packedgenParticles;
  iEvent.getByToken(packedgenParticlesToken, packedgenParticles);

  auto finalParticles = iEvent.getHandle(finalGenParticleToken);

  reco::GenParticleCollection::const_iterator genPart;

  Lepts_RelIso.clear();
  int j = -1;

  for(genPart = finalParticles->begin(); genPart != finalParticles->end(); genPart++){
    j++;
    if(abs(genPart->pdgId())==11  || abs(genPart->pdgId())==13 || abs(genPart->pdgId())==15) {
      if (!(genPart->status()==1 || abs(genPart->pdgId())==15)) {
        Lepts_RelIso.push_back(999);
        continue;
      }
      int ID = MotherID_geniso(&finalParticles->at(j));
      if (!(ID==23 || ID==443 || ID==553 || ID==24)) {
        Lepts_RelIso.push_back(999);
        continue;
      }

      TLorentzVector lep_dressed;
      lep_dressed.SetPtEtaPhiE(genPart->pt(),genPart->eta(),genPart->phi(),genPart->energy());
      std::set<int> gen_fsrset;
      for(size_t k=0; k<packedgenParticles->size();k++){
          if( (*packedgenParticles)[k].status() != 1) continue;
          if( (*packedgenParticles)[k].pdgId() != 22) continue;
          double this_dR_lgamma = reco::deltaR(genPart->eta(), genPart->phi(), (*packedgenParticles)[k].eta(), (*packedgenParticles)[k].phi());
          bool idmatch=false;
          if ((*packedgenParticles)[k].mother(0)->pdgId()==genPart->pdgId() ) idmatch=true;
          const reco::Candidate * mother = (*packedgenParticles)[k].mother(0);
          for(size_t m=0;m<mother->numberOfMothers();m++) {
              if ( (*packedgenParticles)[k].mother(m)->pdgId() == genPart->pdgId() ) idmatch=true;
          }
          if (!idmatch) continue;
          if(this_dR_lgamma<0.3) {
              gen_fsrset.insert(k);
              TLorentzVector gamma;
              gamma.SetPtEtaPhiE((*packedgenParticles)[k].pt(),(*packedgenParticles)[k].eta(),(*packedgenParticles)[k].phi(),(*packedgenParticles)[k].energy());
              lep_dressed = lep_dressed+gamma;
          }
      }
      float this_GENiso = 0.0;
      TLorentzVector thisLep;
      thisLep.SetPtEtaPhiM(lep_dressed.Pt(), lep_dressed.Eta(), lep_dressed.Phi(), lep_dressed.M());
      this_GENiso = computeIso(thisLep, packedgenParticles, gen_fsrset, true);
      Lepts_RelIso.push_back(this_GENiso);
    } else {
        float this_GENiso_nolep=0.0;
        std::set<int> gen_fsrset_nolep;
        TLorentzVector thisPart;
        thisPart.SetPtEtaPhiE(genPart->pt(),genPart->eta(),genPart->phi(),genPart->energy());
        this_GENiso_nolep = computeIso(thisPart, packedgenParticles, gen_fsrset_nolep, false);
        Lepts_RelIso.push_back(this_GENiso_nolep);
    }
  }

  auto isoV = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerIsoMap(*isoV);
  fillerIsoMap.insert(finalParticles, Lepts_RelIso.begin(), Lepts_RelIso.end());
  fillerIsoMap.fill();
  iEvent.put(std::move(isoV));
}

float GenPartIsoProducer::computeIso(TLorentzVector thisPart, edm::Handle<edm::View<pat::PackedGenParticle> > packedgenParticles, std::set<int> gen_fsrset, bool skip_leptons){
  double this_GENiso=0.0;
  for(size_t k=0; k<packedgenParticles->size();k++){
    if( (*packedgenParticles)[k].status() != 1 ) continue;
    if (abs((*packedgenParticles)[k].pdgId())==12 || abs((*packedgenParticles)[k].pdgId())==14 || abs((*packedgenParticles)[k].pdgId())==16) continue;
    if ( abs((*packedgenParticles)[k].pt() - thisPart.Pt())<0.1 && abs((*packedgenParticles)[k].eta() - thisPart.Eta())<0.1 && abs((*packedgenParticles)[k].phi() - thisPart.Phi())<0.1 ) continue;
    if (skip_leptons == true) {
      if ((abs((*packedgenParticles)[k].pdgId())==11 || abs((*packedgenParticles)[k].pdgId())==13)) continue;
      if (gen_fsrset.find(k)!=gen_fsrset.end()) continue;
    }
    double this_dRvL_nolep = reco::deltaR(thisPart.Eta(), thisPart.Phi(), (*packedgenParticles)[k].eta(), (*packedgenParticles)[k].phi());
    if(this_dRvL_nolep<0.3) {
      this_GENiso = this_GENiso + (*packedgenParticles)[k].pt();
    }
  }
  this_GENiso = this_GENiso/thisPart.Pt();
  return this_GENiso;
}

void GenPartIsoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  descriptions.addDefault(desc);
}

  //define this as a plug-in
  DEFINE_FWK_MODULE(GenPartIsoProducer);

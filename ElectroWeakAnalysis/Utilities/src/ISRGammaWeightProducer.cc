#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"
#include "CommonTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>

//
// class declaration
//
class ISRGammaWeightProducer : public edm::EDProducer {
   public:
      explicit ISRGammaWeightProducer(const edm::ParameterSet&);
      ~ISRGammaWeightProducer();

      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

   private:
      edm::InputTag genTag_;
      std::vector<double> isrBinEdges_;
      std::vector<double> ptWeights_;
};


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
/////////////////////////////////////////////////////////////////////////////////////
ISRGammaWeightProducer::ISRGammaWeightProducer(const edm::ParameterSet& pset) {
      genTag_ = pset.getUntrackedParameter<edm::InputTag> ("GenTag", edm::InputTag("generator"));

      produces<double>();
} 

/////////////////////////////////////////////////////////////////////////////////////
ISRGammaWeightProducer::~ISRGammaWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void ISRGammaWeightProducer::beginJob() {}

/////////////////////////////////////////////////////////////////////////////////////
void ISRGammaWeightProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void ISRGammaWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

      if (iEvent.isRealData()) return;

      edm::Handle<reco::GenParticleCollection> genParticles;
      iEvent.getByLabel("genParticles", genParticles);
      unsigned int gensize = genParticles->size();

      std::auto_ptr<double> weight (new double);

      // Set a default weight to start with
      (*weight) = 1.;

      const reco::Candidate* parton = 0;
      const reco::GenParticle* boson = 0;
      const reco::GenParticle* photon = 0;
      for (unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            int partId = abs(part.pdgId());
            if (partId==23 || abs(partId)==24) {
                  boson = &(*genParticles)[i];
            } else if (partId==22) {
                  if (part.numberOfMothers()!=1) continue;
                  const reco::Candidate * mother = part.mother();
                  if (mother->status()!=3) continue;
                  if (abs(mother->pdgId())>6 && mother->pdgId()!=2212) continue;
                  photon = &(*genParticles)[i];
                  parton = mother;
                  if (boson && photon && parton) break;
            } 
            if (boson && photon && parton) break;
      }

      if (boson && photon && parton) {
            double en_s = boson->energy() + photon->energy();
            double px_s = boson->px() + photon->px();
            double py_s = boson->py() + photon->py();
            double pz_s = boson->pz() + photon->pz();
            double s = en_s*en_s - px_s*px_s -py_s*py_s - pz_s*pz_s;
            double en_t = parton->energy() - photon->energy();
            double px_t = parton->px() - photon->px();
            double py_t = parton->py() - photon->py();
            double pz_t = parton->pz() - photon->pz();
            double t = en_t*en_t - px_t*px_t -py_t*py_t - pz_t*pz_t;
            double MV = boson->mass();
            double u = MV*MV - s - t;
            (*weight) = 1. - 2*t*u/(s*s+pow(MV,4));
      }

      iEvent.put(weight);
}

DEFINE_FWK_MODULE(ISRGammaWeightProducer);

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "PhysicsTools/CandUtils/interface/AddFourMomenta.h"
#include "PhysicsTools/CandUtils/interface/Booster.h"
#include <Math/VectorUtil.h>

//
// class declaration
//
class FSRWeightProducer : public edm::EDProducer {
   public:
      explicit FSRWeightProducer(const edm::ParameterSet&);
      ~FSRWeightProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag genTag_;
};


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
/////////////////////////////////////////////////////////////////////////////////////
FSRWeightProducer::FSRWeightProducer(const edm::ParameterSet& pset) {
      genTag_ = pset.getUntrackedParameter<edm::InputTag> ("GenTag", edm::InputTag("generator"));

      produces<double>();
} 

/////////////////////////////////////////////////////////////////////////////////////
FSRWeightProducer::~FSRWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void FSRWeightProducer::beginJob(const edm::EventSetup&) {}

/////////////////////////////////////////////////////////////////////////////////////
void FSRWeightProducer::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void FSRWeightProducer::produce(edm::Event& iEvent, const edm::EventSetup&) {

      if (iEvent.isRealData()) return;

      edm::Handle<reco::GenParticleCollection> genParticles;
      iEvent.getByLabel("genParticles", genParticles);

      std::auto_ptr<double> weight (new double);

      // Set a default weight to start with
      (*weight) = 1.;

      unsigned int gensize = genParticles->size();
      for (unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& lepton = (*genParticles)[i];
            if (lepton.status()!=3) continue;
            int leptonId = lepton.pdgId();
            if (abs(leptonId)!=11 && abs(leptonId)!=13) continue;
            if (lepton.numberOfMothers()!=1) continue;
            const reco::Candidate * boson = lepton.mother();
            int bosonId = boson->pdgId();
            if (bosonId!=23 && abs(bosonId)!=24) continue;
            unsigned int nDaughters = lepton.numberOfDaughters();
            if (nDaughters<=1) continue;
            double leptonMass = lepton.mass();
            double leptonEnergy = lepton.energy();
            double betaLepton = sqrt(1-pow(leptonMass/leptonEnergy,2));
            double bosonMass = boson->mass();
            double cosLeptonTheta = cos(lepton.theta());
            double sinLeptonTheta = sin(lepton.theta());
            double leptonPhi = lepton.phi();
            for (unsigned int j = 0; j<nDaughters; ++j) {
                  const reco::Candidate * photon = lepton.daughter(j);
                  if (photon->pdgId()!=22) continue;
                  double photonEnergy = photon->energy();
                  double cosPhotonTheta = cos(photon->theta());
                  double sinPhotonTheta = sin(photon->theta());
                  double photonPhi = photon->phi();
                  double costheta = sinLeptonTheta*sinPhotonTheta*cos(leptonPhi-photonPhi)
                                  + cosLeptonTheta*cosPhotonTheta;
                  // Missing O(alpha) terms
                  // This is only relevant for W, for Z the effect is negligible
                  if (abs(bosonId)==24) {
                        double delta = - 8*photonEnergy *(1-betaLepton*costheta)
                          / pow(bosonMass,3) 
                          / (1-pow(leptonMass/bosonMass,2))
                          / (4-pow(leptonMass/bosonMass,2))
                          * leptonEnergy * (pow(leptonMass,2)/bosonMass+2*photonEnergy);
                        (*weight) *= (1 + delta);
                  }
                  // Add some extra factor from missing higher orders (in progress)
                  // (*weight) *= ...;
            }
      }


      iEvent.put(weight);
}

DEFINE_FWK_MODULE(FSRWeightProducer);

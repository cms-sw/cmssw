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

      virtual void beginJob() override ;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override ;

   private:
      edm::EDGetTokenT<reco::GenParticleCollection> genToken_;
      std::vector<double> isrBinEdges_;
      std::vector<double> ptWeights_;
};


/////////////////////////////////////////////////////////////////////////////////////
ISRGammaWeightProducer::ISRGammaWeightProducer(const edm::ParameterSet& pset) {
      genToken_ = consumes<reco::GenParticleCollection>(pset.getUntrackedParameter<edm::InputTag> ("GenTag", edm::InputTag("genParticles")));

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
      iEvent.getByToken(genToken_, genParticles);
      unsigned int gensize = genParticles->size();

      std::auto_ptr<double> weight (new double);

      // Set a default weight to start with
      (*weight) = 1.;

      // Find the boson at the hard scattering level
      const reco::GenParticle* boson = 0;
      int parton1Key = -1;
      int parton2Key = -1;
      for (unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            int status = abs(part.status());
            if (status!=3) continue;
            if (part.numberOfMothers()!=2) continue;
            int partId = abs(part.pdgId());
            if (status==3 && (partId==23||abs(partId)==24)) {
                  boson = &(*genParticles)[i];
                  parton1Key = part.motherRef(0).key();
                  parton2Key = part.motherRef(1).key();
                  break;
            }
      }

      // Consider only photons near the hard-scattering process
      const reco::GenParticle* photon = 0;
      if (boson) {
        for (unsigned int i = 0; i<gensize; ++i) {
            photon = 0;
            const reco::GenParticle& part = (*genParticles)[i];
            int status = abs(part.status());
            if (status!=1) continue;
            int partId = abs(part.pdgId());
            if (partId!=22)  continue;
            if (part.numberOfMothers()!=1) continue;
            int keyM = part.motherRef(0).key();
            const reco::GenParticle* mother = &(*genParticles)[keyM];
            if (mother->status()!=3) continue;
            int mId = mother->pdgId();
            if (abs(mId)>6 && mId!=2212) continue;
            for (unsigned int j=0; j<mother->numberOfDaughters(); ++j){
                  int keyD = mother->daughterRef(j).key();
                  if (keyD==parton1Key || keyD==parton2Key) {
                        photon = &part;
                        break;
                  }
            }
            if (photon) break;
        }
      }

      if (boson && photon) {
            math::XYZTLorentzVector smom = boson->p4() + photon->p4();
            double s = smom.M2();
            double sqrts = smom.M();

            // Go to CM using the boost direction of the boson+photon system
            ROOT::Math::Boost cmboost(smom.BoostToCM());
            math::XYZTLorentzVector photonCM(cmboost(photon->p4()));
            double pcostheta = (  smom.x()*photonCM.x()
                               + smom.y()*photonCM.y()
                               + smom.z()*photonCM.z() ) / smom.P();

            // Determine kinematic invariants
            double t = - sqrts * (photonCM.t()-pcostheta);
            double MV = boson->mass();
            double u = MV*MV - s - t;
            (*weight) = 1. - 2*t*u/(s*s+MV*MV*MV*MV);
            //printf(">>>>>>>>> s %f t %f u %f, MV %f, weight = %f\n", s, t, u, MV, (*weight));
      }

      iEvent.put(weight);
}

DEFINE_FWK_MODULE(ISRGammaWeightProducer);

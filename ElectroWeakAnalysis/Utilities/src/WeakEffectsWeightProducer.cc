////////// Header section /////////////////////////////////////////////
#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class WeakEffectsWeightProducer: public edm::EDProducer {
public:
      WeakEffectsWeightProducer(const edm::ParameterSet& pset);
      virtual ~WeakEffectsWeightProducer();
      virtual void beginJob() override ;
      virtual void produce(edm::Event &, const edm::EventSetup&) override;
      virtual void endJob() override ;
private:
      edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;
      double rhoParameter_;

      double alphaQED(double q2);
      double sigma0_qqbarll(unsigned int quark_type, double Q, double rho);

};

////////// Source code ////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

/////////////////////////////////////////////////////////////////////////////////////
WeakEffectsWeightProducer::WeakEffectsWeightProducer(const edm::ParameterSet& pset) :
//       genParticlesToken_(consumes<reco::GenParticleCollection>(pset.getUntrackedParameter<edm::InputTag> ("GenParticlesTag", edm::InputTag("genParticles")))),
      genParticlesToken_(consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"))),
      rhoParameter_(pset.getUntrackedParameter<double> ("RhoParameter", 1.004))
{
      produces<double>();
}

/////////////////////////////////////////////////////////////////////////////////////
WeakEffectsWeightProducer::~WeakEffectsWeightProducer(){}

/////////////////////////////////////////////////////////////////////////////////////
void WeakEffectsWeightProducer::beginJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void WeakEffectsWeightProducer::endJob(){
}

/////////////////////////////////////////////////////////////////////////////////////
void WeakEffectsWeightProducer::produce(edm::Event & iEvent, const edm::EventSetup&){
      if (iEvent.isRealData()) return;

      edm::Handle<reco::GenParticleCollection> genParticles;
      iEvent.getByToken(genParticlesToken_, genParticles);
      unsigned int gensize = genParticles->size();

      std::auto_ptr<double> weight (new double);

      // Set default weight
      (*weight) = 1.;

      // Only DY implemented for the time being
      for(unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            int status = part.status();
            if (status!=3) break;
            int id = part.pdgId();
            if (id!=23) continue;
            double Q = part.mass();
            unsigned int nmothers = part.numberOfMothers();
            if (nmothers<=0) continue;
            size_t key = part.motherRef(0).key();
            unsigned int quark_id = abs((*genParticles)[key].pdgId());
            if (quark_id>0 && quark_id<6) {
                  (*weight) *= sigma0_qqbarll(quark_id, Q, rhoParameter_)
                             / sigma0_qqbarll(quark_id, Q, 1.0);
            }
            break;
      }

      //printf(" \t >>>>> WeakEffectsWeightProducer: Final weight = %f\n", (*weight));
      iEvent.put(weight);
}

double WeakEffectsWeightProducer::alphaQED(double q2) {
      double pigaga = -0.010449239475366825 - 0.0023228196282246765*log(q2)- 0.0288 - 0.002980*(log(q2/8464.)+0.006307*(q2/8464.-1.));
      return (1./137.0359895) / (1.+pigaga);
}

double WeakEffectsWeightProducer::sigma0_qqbarll(unsigned int quark_id, double Q, double rho) {
      double MZ = 91.188;
      double GZ = 2.495;
      double sin2eff = 0.232;

      double vl = -0.5 + 2.*sin2eff;
      double al = -0.5;

      double qq = 0.;
      double vq = 0.;
      double aq = 0.;
      double alphaW = 2.7e-3 * pow(log(Q*Q/80.4/80.4),2);
      double alphaZ = 2.7e-3 * pow(log(Q*Q/MZ/MZ),2);
      double sudakov_factor = 1.;
      if (abs(quark_id)%2==1) {
            qq = -1./3.;
            vq = -0.5 - 2.*qq*sin2eff;
            aq = -0.5;
            sudakov_factor = 1 + (-2.139 + 0.864)*alphaW - 0.385*alphaZ;
      } else {
            qq = 2./3.;
            vq = 0.5 - 2.*qq*sin2eff;
            aq = 0.5;
            sudakov_factor = 1 + (-3.423 + 1.807)*alphaW - 0.557*alphaZ;
      }

      double alfarn = alphaQED(Q*Q);
      double zcoupl = sqrt(2.) * 1.166389e-5 * MZ*MZ / 4. / M_PI;
      double gll = zcoupl * MZ/3. * (vl*vl + al*al);
      double gdd = zcoupl * MZ/3. * (vq*vq + aq*aq);
      double denom =  (Q*Q-MZ*MZ)*(Q*Q-MZ*MZ)+ pow(Q,4)*GZ*GZ/MZ/MZ;
      double qed = M_PI * qq*qq * alfarn*alfarn / Q/Q;
      double zint = rho * 2*M_PI * zcoupl * alfarn * qq * vq*vl * (Q*Q-MZ*MZ) / denom;
      double zonly = rho * rho * 9.*M_PI * gll * gdd / MZ/MZ * Q*Q / denom;

      return (qed + zint + zonly) * sudakov_factor;
}

DEFINE_FWK_MODULE(WeakEffectsWeightProducer);

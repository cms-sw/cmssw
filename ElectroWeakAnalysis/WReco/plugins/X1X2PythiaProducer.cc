//
// Original Author:  Juan Alcaraz, April 2008
//
//


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
class X1X2PythiaProducer : public edm::EDProducer {
   public:
      explicit X1X2PythiaProducer(const edm::ParameterSet&);
      ~X1X2PythiaProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      bool FindX1X2(const reco::Candidate* gen1, const reco::Candidate* gen2);
      double Mass12(const reco::Candidate* ge1, const reco::Candidate* ge2);
      double pt2Evol(const reco::Candidate* ge1, const reco::Candidate* ge2, int choice);
      double _x1;
      double _x2;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
X1X2PythiaProducer::X1X2PythiaProducer(const edm::ParameterSet& iConfig) {
   //register your products
   produces<std::vector<double> >();
   produces<std::vector<int> >();
}
X1X2PythiaProducer::~X1X2PythiaProducer(){}

//
// member functions
//
// ------------ method called to produce the data  ------------
void
X1X2PythiaProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

      // New method (CMSSW versions > 167)
      bool newgen = true;
      edm::Handle<reco::GenParticleCollection> genNewParticles;
      // Old method (CMSSW versions <= 167)
      edm::Handle<reco::CandidateCollection> genParticles;

      try {
            iEvent.getByLabel("genParticles", genNewParticles);
      } catch (...) {
            // Try old method if new one did not work
            newgen = false;
            iEvent.getByLabel("genParticleCandidates", genParticles);
      }

      //DumpEvent(iEvent);

      const reco::Candidate* m1 = 0;
      const reco::Candidate* m2 = 0;

      unsigned int gensize = 0;
      if (newgen) {
            gensize = genNewParticles->size();
            
      } else {
            gensize = genParticles->size();
      }
      for(unsigned int i = 0; i<gensize; ++i) {
            const reco::Candidate* part;
            if (newgen) {
                  part = &((*genNewParticles)[i]);
            } else {
                  part = &((*genParticles)[i]);
            }
            int status = part->status();
            if (status!=3) continue;
            int id = part->pdgId();
            if (id != 23 && abs(id) != 24 ) continue;
            int nmothers = part->numberOfMothers();
            if (nmothers!=2) continue;
            m1 = part->mother(0);
            m2 = part->mother(1);
            break;
      }

      if (m1 && m2 && FindX1X2(m1, m2) ) {
            std::auto_ptr<std::vector<double> > xfrac (new std::vector<double>);
            xfrac->push_back(_x1);
            xfrac->push_back(_x2);
            iEvent.put(xfrac);
            std::auto_ptr<std::vector<int> > iflav (new std::vector<int>);
            iflav->push_back(m1->pdgId());
            iflav->push_back(m2->pdgId());
            iEvent.put(iflav);
            if (newgen) {
                  LogTrace("") << ">>> Parton 1: flavor = " << m1->pdgId() << ", x1 = " << _x1 << ", naive x1= " << fabs(m1->pz()/((*genNewParticles)[0]).pz());
                  LogTrace("") << ">>> Parton 2: flavor = " << m2->pdgId() << ", x2 = " << _x2 << ", naive x2= " << fabs(m2->pz()/((*genNewParticles)[0]).pz());
            } else {
                  LogTrace("") << ">>> Parton 1: flavor = " << m1->pdgId() << ", x1 = " << _x1 << ", naive x1= " << fabs(m1->pz()/((*genParticles)[0]).pz());
                  LogTrace("") << ">>> Parton 2: flavor = " << m2->pdgId() << ", x2 = " << _x2 << ", naive x2= " << fabs(m2->pz()/((*genParticles)[0]).pz());
            }
      }
}
      
bool X1X2PythiaProducer::FindX1X2(const reco::Candidate* gen1, const reco::Candidate* gen2) {
      _x1 = 1.;
      _x2 = 1.;

      // gen1 and gen2 are the links to the partons 
      // before the hard scattering process
      if (!gen1) return false;
      if (!gen2) return false;

      const reco::Candidate* g1 = gen1;
      const reco::Candidate* g2 = gen2;
      const reco::Candidate* gprev1 = NULL;
      const reco::Candidate* gprev2 = NULL;
      while (1) {
            if (g1==gprev1 && g2==gprev2) break;
            gprev1 = g1;
            gprev2 = g2;
            double sold = Mass12(g1,g2);

            bool choose_x1 = false;
            if (g1->numberOfMothers()>0 && g2->numberOfMothers()>0) {
                  double pt12 = pt2Evol(g1, g2, 1);
                  double pt22 = pt2Evol(g1, g2, 2);

                  if (pt12 > pt22) choose_x1 = true; 
                  else choose_x1 = false;

                  //double mass12 = Mass12(g1->mother(),g2);
                  //double mass22 = Mass12(g1,g2->mother());
                  //printf("sold %f snew1 %f snew2 %f\n", sold, mass12, mass22);
                  //printf("z1 %f z2 %f pt12 %f pt22 %f\n", sold/mass12, sold/mass22, pt12, pt22);
            } else if (g1->numberOfMothers()>0) {
                  choose_x1 = true;
            } else if (g2->numberOfMothers()>0) {
                  choose_x1 = false;
            } else {
                  continue;
            }
            if (choose_x1) {
                  g1 = g1->mother();
                  _x1 *= sold/Mass12(g1,g2);
                  //printf("x1_step %.3f\n", sold/Mass12(g1,g2));
            } else {
                  g2 = g2->mother();
                  _x2 *= sold/Mass12(g1,g2);
                  //printf("x2_step %.3f\n", sold/Mass12(g1,g2));
            }
      }

      return true;
}

// ------------ method called once each job just before starting event loop  ------------
void 
X1X2PythiaProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
X1X2PythiaProducer::endJob() {
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double X1X2PythiaProducer::Mass12(const reco::Candidate* ge1, const reco::Candidate* ge2){
      double en = ge1->energy() + ge2->energy();
      double px = ge1->px() + ge2->px();
      double py = ge1->py() + ge2->py();
      double pz = ge1->pz() + ge2->pz();
      return  en*en - px*px -py*py - pz*pz;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double X1X2PythiaProducer::pt2Evol(const reco::Candidate* ge1, const reco::Candidate* ge2, int choice){

      const reco::Candidate* gea = ge1->mother();
      const reco::Candidate* geb = ge1;
      const reco::Candidate* ger = ge2;
      if (choice!=1) {
            gea = ge2->mother();
            geb = ge2;
            ger = ge1;
      }

      reco::CompositeCandidate geBR;
      geBR.addDaughter(*geb);
      geBR.addDaughter(*ger);
      AddFourMomenta addP4;
      addP4.set(geBR);

      reco::Candidate* cb = geb->clone();
      math::XYZVector boost(geBR.boostToCM());
      cb->setP4(ROOT::Math::VectorUtil::boost(cb->p4(),boost));
      //boost.set(*cb);

      double mBR = geBR.mass();
      double px = cb->px()/cb->p()/2/mBR;
      double py = cb->py()/cb->p()/2/mBR;
      double pz = cb->pz()/cb->p()/2/mBR;
      double en = -1./2/mBR;
      math::XYZTLorentzVectorD p4aux(px,py,pz,en);
      //reco::GenParticle caux(cb->charge(), p4aux, cb->pdgId(), cb->status(), cb->charge(), false);
      //boost.Invert();
      //caux.setP4(ROOT::Math::VectorUtil::boost(caux.p4(),boost);
      cb->setP4(p4aux);
      cb->setP4(ROOT::Math::VectorUtil::boost(cb->p4(),-boost));

      delete cb;

      double eA = gea->energy();
      double pxA = gea->px();
      double pyA = gea->py();
      double pzA = gea->pz();

      double eB = geb->energy();
      double pxB = geb->px();
      double pyB = geb->py();
      double pzB = geb->pz();

      double eAUX = cb->energy();
      double pxAUX = cb->px();
      double pyAUX = cb->py();
      double pzAUX = cb->pz();

      double papb = eA*eB - pxA*pxB - pyA*pyB - pzA*pzB;
      double papAUX = eA*eAUX - pxA*pxAUX - pyA*pyAUX - pzA*pzAUX;
      //printf("choice %d papb %f papAUX %f\n", choice, papb, papAUX);

      double Q2 = -papb / (papAUX + 0.5);
      double z = mBR/Mass12(gea,ger);
      if (papAUX>=-0.50) {
            // This should not happen
            LogTrace("") << ">>> WARNING: unstable calculation; papAUX = " << papAUX << "; ad-hoc fix by setting papAUX=-0,501";
            papAUX = -0.501;
      }
      double pt2 = (1-z)*Q2;
      if (pt2<-0.1 && z>0.02) {
            // pt2<0 can happen due to accuracy problem when:
            //    * z is too small: z<0.01 or so
            // Otherwise one should worry about, so dump it out...
            LogTrace("") << ">>> WARNING: unstable calculation; pt2Evol = " << pt2 << ", z = " << z << ", papb = " << papb << "papAUX = " << papAUX;
      }
      return pt2;
}

//define this as a plug-in
DEFINE_FWK_MODULE(X1X2PythiaProducer);

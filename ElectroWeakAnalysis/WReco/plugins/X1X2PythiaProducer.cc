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
      double Pt2(const reco::Candidate* ge1, const reco::Candidate* geref);
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
                  double pt12 = Pt2(g1, g1->mother());
                  double pt22 = Pt2(g2, g2->mother());

                  // Use a more appropriate pt2 for the evolution
                  double mass12 = Mass12(g1->mother(),g2);
                  double z1 = sold / mass12;
                  double fact1 = (1-z1)*(1-z1)*mass12/2;
                  pt12 = fact1*(1-sqrt(1-2*pt12/fact1));

                  // Use a more appropriate pt2 for the evolution
                  double mass22 = Mass12(g1,g2->mother());
                  double z2 = sold / mass22;
                  double fact2 = (1-z2)*(1-z2)*mass22/2;
                  pt22 = fact2*(1-sqrt(1-2*pt22/fact2));

                  if (pt12 > pt22) choose_x1 = true; 
                  else choose_x1 = false;

                  //printf("sold %f snew1 %f snew2 %f\n", sold, mass12, mass22);
                  //printf("z1 %f z2 %f pt12 %f pt22 %f\n", z1, z2, pt12, pt22);
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
double X1X2PythiaProducer::Pt2(const reco::Candidate* ge1, const reco::Candidate* geref){
      double p1proj = (ge1->px()*geref->px()+ge1->py()*geref->py()+ge1->pz()*geref->pz()) / geref->p();
      return ge1->p()*ge1->p() - p1proj*p1proj;
}

//define this as a plug-in
DEFINE_FWK_MODULE(X1X2PythiaProducer);

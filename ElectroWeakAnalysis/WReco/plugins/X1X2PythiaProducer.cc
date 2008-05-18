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

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

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
      bool FindX1X2(edm::Handle<reco::GenParticleCollection>, const reco::GenParticle*, const reco::GenParticle*);
      double Mass12(const reco::GenParticle*, const reco::GenParticle*);
      double Q2InBranch(edm::Handle<reco::GenParticleCollection>, const reco::GenParticle*);
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
   //produces<reco::GenParticleCollection>();
}
X1X2PythiaProducer::~X1X2PythiaProducer(){}

//
// member functions

//
// ------------ method called to produce the data  ------------
void
X1X2PythiaProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

      edm::Handle<reco::GenParticleCollection> genParticles;
      iEvent.getByLabel("genParticles", genParticles);
      unsigned int gensize = genParticles->size();

      const reco::GenParticle* m1 = 0;
      const reco::GenParticle* m2 = 0;

      for(unsigned int i = 0; i<gensize; ++i) {
            const reco::GenParticle& part = (*genParticles)[i];
            //printf("status %d id %d\n", part.status(), part.pdgId());
            int status = part.status();
            if (status!=3) continue;
            int id = part.pdgId();
            if (id != 23 && abs(id) != 24 ) continue;
            int nmothers = part.numberOfMothers();
            //printf("Boson Id %d numberOfMothers %d\n", id, nmothers);
            if (nmothers!=2) continue;
            size_t key1 = part.motherRef(0).key();
            m1 = &((*genParticles)[key1]);
            size_t key2 = part.motherRef(1).key();
            m2 = &((*genParticles)[key2]);
            //printf("m1 %d m2 %d\n", m1->pdgId(), m2->pdgId());
            break;
      }

      if (m1 && m2 && FindX1X2(genParticles, m1, m2) ) {
            std::auto_ptr<std::vector<double> > xfrac (new std::vector<double>);

            xfrac->push_back(_x1);
            xfrac->push_back(_x2);
            iEvent.put(xfrac);
            std::auto_ptr<std::vector<int> > iflav (new std::vector<int>);
            iflav->push_back(m1->pdgId());
            iflav->push_back(m2->pdgId());
            iEvent.put(iflav);
            LogTrace("") << "Naive:      flv1 " << m1->pdgId() << " x1 " << (m1->energy()+fabs(m1->pz()))/2/(*genParticles)[0].pz()<< " flv2 " << m2->pdgId() << " x2 " << (m2->energy()+fabs(m2->pz()))/2/(*genParticles)[0].pz();
            LogTrace("") << "X1X2PyProd: flv1 " << m1->pdgId() << " x1 " << _x1 << " flv2 " << m2->pdgId() << " x2 " << _x2;

            if (edm::MessageDrop::instance()->debugEnabled) {
              try {
                  edm::Handle<edm::HepMCProduct> mc;
                  iEvent.getByLabel("source",mc);
                  const HepMC::GenEvent* genev = mc->GetEvent();
                  HepMC::PdfInfo* pdfstuff = genev->pdf_info();
                  if (pdfstuff!=0) {
                        LogTrace("") << "PDFInfo:    flv1 " << pdfstuff->id1() << " x1 " << pdfstuff->x1() << " flv2 " << pdfstuff->id2() << " x2 " << pdfstuff->x2();
                        LogTrace("") << "PDFInfo:   Scale " << pdfstuff->scalePDF();
                        //genev->print();
                  }
              } catch (...) {
              }
            }

      }

}
      
bool X1X2PythiaProducer::FindX1X2(edm::Handle<reco::GenParticleCollection> genParticles, const reco::GenParticle* gen1, const reco::GenParticle* gen2) {
      _x1 = 1.;
      _x2 = 1.;

      // gen1 and gen2 are the links to the partons 
      // before the hard scattering process
      if (!gen1) return false;
      if (!gen2) return false;

      const reco::GenParticle* g1 = gen1;
      const reco::GenParticle* g2 = gen2;
      const reco::GenParticle* gprev1 = NULL;
      const reco::GenParticle* gprev2 = NULL;
      while (1) {
            if (g1==gprev1 && g2==gprev2) break;
            gprev1 = g1;
            gprev2 = g2;
            double sold = Mass12(g1,g2);
            const reco::GenParticle* m1 = 0;
            const reco::GenParticle* m2 = 0;

            bool choose_x1 = false;
            if (g1->numberOfMothers()>0 && g2->numberOfMothers()>0) {
                  double Q21 = Q2InBranch(genParticles, g1);
                  double Q22 = Q2InBranch(genParticles, g2);
                  size_t key1 = g1->motherRef(0).key();
                  size_t key2 = g2->motherRef(0).key();
                  m1 = &((*genParticles)[key1]);
                  m2 = &((*genParticles)[key2]);
                  double mass12 = Mass12(m1,g2);
                  double mass22 = Mass12(g1,m2);
                  double z1 = sold/mass12;
                  double z2 = sold/mass22;
                  double pt21 = (1-z1)*Q21;
                  double pt22 = (1-z2)*Q22;

                  if (pt21 > pt22) choose_x1 = true; 
                  else choose_x1 = false;

            } else if (g1->numberOfMothers()>0) {
                  choose_x1 = true;
                  size_t key1 = g1->motherRef(0).key();
                  m1 = &((*genParticles)[key1]);
            } else if (g2->numberOfMothers()>0) {
                  choose_x1 = false;
                  size_t key2 = g2->motherRef(0).key();
                  m2 = &((*genParticles)[key2]);
            } else {
                  continue;
            }

            if (choose_x1) {
                  double mass12 = Mass12(m1,g2);
                  _x1 *= sold/mass12;
                  if (edm::MessageDrop::instance()->debugEnabled) {
                        double q21 = Q2InBranch(genParticles, g1);
                        LogTrace("") << "x1_step " << sold/mass12 << " Q21 " <<  q21 << " pt21 " << (1-sold/mass12)*q21 << " sold " << sold << " mass12 " << mass12;
                        if (g2->numberOfMothers()>0) {
                              double mass22 = Mass12(g1,m2);
                              double q22 = Q2InBranch(genParticles, g2);
                              LogTrace("") << "... NOT SELECTED: x2_step " << sold/mass22 << " Q22 " <<  q22 << " pt22 " << (1-sold/mass22)*q22 << " sold " << sold << " mass22 " << mass22;
                        }
                  }
                  g1 = m1;
            } else {
                  double mass22 = Mass12(g1,m2);
                  _x2 *= sold/mass22;
                  if (edm::MessageDrop::instance()->debugEnabled) {
                        double q22 = Q2InBranch(genParticles, g2);
                        LogTrace("") << "x2_step " << sold/mass22 << " Q22 " <<  q22 << " pt22 " << (1-sold/mass22)*q22 << " sold " << sold << " mass22 " << mass22;
                        if (g1->numberOfMothers()>0) {
                              double mass12 = Mass12(m1,g2);
                              double q21 = Q2InBranch(genParticles, g1);
                              LogTrace("") << "... NOT SELECTED: x1_step " << sold/mass12 << " Q21 " <<  q21 << " pt21 " << (1-sold/mass12)*q21 << " sold " << sold << " mass12 " << mass12;
                        }
                  }
                  g2 = m2;
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
double X1X2PythiaProducer::Mass12(const reco::GenParticle* ge1, const reco::GenParticle* ge2){
      double en = ge1->energy() + ge2->energy();
      double px = ge1->px() + ge2->px();
      double py = ge1->py() + ge2->py();
      double pz = ge1->pz() + ge2->pz();
      return  en*en - px*px -py*py - pz*pz;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
double X1X2PythiaProducer::Q2InBranch(edm::Handle<reco::GenParticleCollection> genParticles, const reco::GenParticle* gp){
      // Initial value set by PYTHIA
      double q2 = gp->p4().M2();

      // Do nothing in some cases (just a protection)
      if (gp->status()!=3 || gp->numberOfMothers()!=1) return q2;

      size_t key = gp->motherRef(0).key();
      const reco::GenParticle* moth = &((*genParticles)[key]);
      //printf("Mother pdgId %d\n", moth->pdgId());
      double px = moth->px();
      double py = moth->py();
      double pz = moth->pz();
      double en = moth->energy();
      //printf("Mother px %f py %f pz %f en %f\n", moth->px(), moth->py(), moth->pz(), moth->energy());
      //printf("Number of daughters %d\n", moth->numberOfDaughters());
      for (unsigned int j=0; j<moth->numberOfDaughters(); ++j) {
            size_t key = moth->daughterRef(j).key();
            const reco::GenParticle* daug = &((*genParticles)[key]);
            if (daug->status()==3) continue; // this is the entry that must be modified...
            px -= daug->px();
            py -= daug->py();
            pz -= daug->pz();
            en -= daug->energy();
            //printf("Daughter px %f py %f pz %f en %f\n", daug->px(), daug->py(), daug->pz(), daug->energy());
      }
      //printf("Q2 BEFORE: %f\n", q2);
      q2 = - en*en + px*px + py*py + pz*pz;
      //printf("Q2 AFTER:  %f\n", q2);

      return q2;
}

//define this as a plug-in
DEFINE_FWK_MODULE(X1X2PythiaProducer);

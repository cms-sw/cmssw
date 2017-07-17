#ifndef gen_TauolaInterface_TauolappInterface_h
#define gen_TauolaInterface_TauolappInterface_h

#include "HepPDT/ParticleDataTable.hh"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterfaceBase.h"
#include "TLorentzVector.h"
#include "TVector.h"


namespace HepMC 
{
class GenEvent;
}

namespace CLHEP
{
class HepRandomEngine;
}

namespace gen {
  extern "C" {
    void ranmar_( float *rvec, int *lenv );
    void rmarin_( int*, int*, int* );
  }

   class TauolappInterface : public TauolaInterfaceBase {
      public:
      
      // ctor & dtor
     TauolappInterface( const edm::ParameterSet& );
     ~TauolappInterface();
     
      void enablePolarization()  { fPolarization = true; return; }
      void disablePolarization() { fPolarization = false; return; }
      void init( const edm::EventSetup& );
      const std::vector<int>& operatesOnParticles() { return fPDGs; }
      HepMC::GenEvent* decay( HepMC::GenEvent* );
      void statistics() ;
      virtual void SetLHE(lhef::LHEEvent *l){lhe=l;}
      void setRandomEngine(CLHEP::HepRandomEngine* v) { fRandomEngine = v; }
      static double flat();

      private: 
      // member function(s)
      void decodeMDTAU( int );
      void selectDecayByMDTAU();
      int selectLeptonic();
      int selectHadronic();

      HepMC::GenEvent*    make_simple_tau_event(const TLorentzVector &l,int pdgid,int status);
      void                update_particles(HepMC::GenParticle* partHep,HepMC::GenEvent* theEvent,HepMC::GenParticle* p,TVector3 &boost);
      bool                isLastTauInChain(const HepMC::GenParticle* tau);
      HepMC::GenParticle* GetMother(HepMC::GenParticle* tau);
      double MatchedLHESpinUp(HepMC::GenParticle* tau, std::vector<HepMC::GenParticle> &p, std::vector<double> &spinup,std::vector<int> &m_idx);
      HepMC::GenParticle* FirstTauInChain(HepMC::GenParticle* tau);
      void BoostProdToLabLifeTimeInDecays(HepMC::GenParticle* p,TLorentzVector &lab, TLorentzVector &prod);

      //
      static CLHEP::HepRandomEngine*           fRandomEngine;            
      std::vector<int>                         fPDGs;
      bool                                     fPolarization;      
      edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;
      edm::ParameterSet*                       fPSet;
      bool                                     fIsInitialized;
      
      int                                      fMDTAU;
      bool                                     fSelectDecayByEvent;
      std::vector<int>                         fLeptonModes;
      std::vector<int>                         fHadronModes;
      std::vector<double>                      fScaledLeptonBrRatios;
      std::vector<double>                      fScaledHadronBrRatios;
      lhef::LHEEvent *lhe;

      double dmMatch;
      bool   dolhe;
      bool   dolheBosonCorr;
      int    ntries;
      double lifetime;
   };

}

#endif

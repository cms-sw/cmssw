// -*- C++ -*-
//
// Package:    FourLepFilter
// Class:      FourLepFilter
// 
/**\class FourLepFilter FourLepFilter.cc psi2s1s/FourLepFilter/src/FourLepFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  bian jianguo
//         Created:  Tue Nov 22 20:39:54 CST 2011
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <iostream>

//
// class declaration
//

class FourLepFilter : public edm::EDFilter {
   public:
      explicit FourLepFilter(const edm::ParameterSet&);
      ~FourLepFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
    
       edm::EDGetToken token_;
       double minPt;
       double maxEta;
       double maxPt;
       double minEta;
       int    particleID;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FourLepFilter::FourLepFilter(const edm::ParameterSet& iConfig):
token_(consumes<edm::HepMCProduct>(edm::InputTag(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")),"unsmeared"))),
minPt(iConfig.getUntrackedParameter("MinPt", 0.)),
maxEta(iConfig.getUntrackedParameter("MaxEta", 10.)),
maxPt(iConfig.getUntrackedParameter("MaxPt", 1000.)),
minEta(iConfig.getUntrackedParameter("MinEta", 0.)),
particleID(iConfig.getUntrackedParameter("ParticleID", 0))
{
}


FourLepFilter::~FourLepFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
FourLepFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
//   FourMuonInput++;
//   std::cout<<"NumberofInputEvent "<<FourMuonInput<<std::endl;

   bool accepted = false;
//   int n4muon = 0;
   int nLeptons = 0;
    
   Handle< HepMCProduct > evt;
   iEvent.getByToken(token_, evt);
   const HepMC::GenEvent * myGenEvent = evt->GetEvent();


   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
      if ( (*p)->status()!=1 ) continue;
//      if ( abs((*p)->pdg_id()) == particleID  )n4muon++;
          if ((*p)->momentum().perp() > minPt && fabs((*p)->momentum().eta()) < maxEta &&
              (*p)->momentum().perp() < maxPt && fabs((*p)->momentum().eta()) > minEta) {
                if ( abs((*p)->pdg_id()) == particleID  )  nLeptons++;
          }
          if (nLeptons >= 4) {
            accepted = true;
//            FourMuonFilter++;
//            std::cout<<"NumberofFourMuonFilter "<<FourMuonFilter<<std::endl;
                break;
          }
   }

//   if(n4muon>=4){FourMuon++; std::cout<<"NumberofFourMuon "<<FourMuon<<std::endl;}

   if (accepted) {
        return true;
   } else {
        return false;
   }


}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
FourLepFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(FourLepFilter);

// -*- C++ -*-
//
// Package:    DJpsiFilter
// Class:      DJpsiFilter
// 
/**\class DJpsiFilter DJpsiFilter.cc psi2s1s/DJpsiFilter/src/DJpsiFilter.cc

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

class DJpsiFilter : public edm::EDFilter {
   public:
      explicit DJpsiFilter(const edm::ParameterSet&);
      ~DJpsiFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:

      virtual bool filter(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------
    
       edm::EDGetToken token_;
       double minPt;
       double maxY;
       double maxPt;
       double minY;
       int    status;
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
DJpsiFilter::DJpsiFilter(const edm::ParameterSet& iConfig):
token_(consumes<edm::HepMCProduct>(iConfig.getUntrackedParameter("moduleLabel",std::string("generator")))),
minPt(iConfig.getUntrackedParameter("MinPt", 0.)),
maxY(iConfig.getUntrackedParameter("MaxY", 10.)),
maxPt(iConfig.getUntrackedParameter("MaxPt", 1000.)),
minY(iConfig.getUntrackedParameter("MinY", 0.)),
status(iConfig.getUntrackedParameter("Status", 0)),
particleID(iConfig.getUntrackedParameter("ParticleID", 0))
{
   //now do what ever initialization is needed
}


DJpsiFilter::~DJpsiFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
DJpsiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
//   DJpsiInput++;
//   std::cout<<"NumberofInputEvent "<<DJpsiInput<<std::endl;

   bool accepted = false;
   int n2jpsi = 0;
//   int n2hadron = 0;
   double energy, pz, momentumY;
   Handle< HepMCProduct > evt;
   iEvent.getByToken(token_, evt);
   const HepMC::GenEvent * myGenEvent = evt->GetEvent();


   for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {
      if ( (*p)->status()!=status ) continue;
//      if ( abs((*p)->pdg_id()) == particleID  )n2hadron++;
      energy=(*p)->momentum().e();
      pz=(*p)->momentum().pz();
      momentumY=0.5*log((energy+pz)/(energy-pz));
          if ((*p)->momentum().perp() > minPt && fabs(momentumY) < maxY &&
              (*p)->momentum().perp() < maxPt && fabs(momentumY) > minY) {
                if ( abs((*p)->pdg_id()) == particleID  )  n2jpsi++;
          }
          if (n2jpsi >= 2) {
            accepted = true;
                break;
          }
   }


   if (accepted) {
        return true;
   } else {
        return false;
   }


}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DJpsiFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(DJpsiFilter);

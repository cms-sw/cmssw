// -*- C++ -*-
//
// Package:    HighMultiplicityGenFilter
// Class:      HighMultiplicityGenFilter
// 
/**\class HighMultiplicityGenFilter HighMultiplicityGenFilter.cc davidlw/HighMultiplicityGenFilter/src/HighMultiplicityGenFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Wei Li
//         Created:  Tue Dec  8 23:51:37 EST 2009
// $Id: HighMultiplicityGenFilter.cc,v 1.1 2010/08/31 21:39:08 edwenger Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//
// class declaration
//

class HighMultiplicityGenFilter : public edm::EDFilter {
   public:
      explicit HighMultiplicityGenFilter(const edm::ParameterSet&);
      ~HighMultiplicityGenFilter();

   private:
      virtual void beginJob();
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      
      // ----------member data ---------------------------
      edm::ESHandle <ParticleDataTable> pdt; 
      double etaMax;
      double ptMin;
      int nMin; 
      int nAccepted;
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
HighMultiplicityGenFilter::HighMultiplicityGenFilter(const edm::ParameterSet& iConfig) :
etaMax(iConfig.getUntrackedParameter<double>("etaMax")),
ptMin(iConfig.getUntrackedParameter<double>("ptMin")),
nMin(iConfig.getUntrackedParameter<int>("nMin"))
{
  //now do what ever initialization is needed
  nAccepted = 0; 
}


HighMultiplicityGenFilter::~HighMultiplicityGenFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HighMultiplicityGenFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  bool accepted = false;
  edm::Handle<edm::HepMCProduct> evt;
  iEvent.getByLabel("generator", evt);

  iSetup.getData(pdt);

  const HepMC::GenEvent * myGenEvent = evt->GetEvent();

  int nMult=0;
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin();   p != myGenEvent->particles_end(); ++p ) {

    if((*p)->status()!=1) continue;

    double charge = 0;
    int pid = (*p)->pdg_id();
    if(abs(pid) > 100000) { std::cout<<"pid="<<pid<<" status="<<(*p)->status()<<std::endl; continue; }
    const ParticleData* part = pdt->particle(pid);
    if(part) charge = part->charge();
    if(charge == 0) continue;

    if ( 
	 (*p)->momentum().perp() > ptMin 
	 && fabs((*p)->momentum().eta()) < etaMax  ) nMult++;
  }
  if(nMult>=nMin) { nAccepted++; accepted = true; }
  return accepted;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HighMultiplicityGenFilter::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
HighMultiplicityGenFilter::endJob() {
  std::cout<<"There are "<<nAccepted<<" events with multiplicity greater than "<<nMin<<std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HighMultiplicityGenFilter);

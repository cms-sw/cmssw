// -*- C++ -*-
//
// Package:    HcalHPDFilter
// Class:      HcalHPDFilter
// 
/**\class HcalHPDFilter HcalHPDFilter.cc RecoHcal/HcalHPDFilter/src/HcalHPDFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Efe Yazgan
//         Created:  Thu Jul 24 10:31:07 CEST 2008
// $Id: HcalHPDFilter.cc,v 1.3 2013/02/27 20:17:14 wmtan Exp $
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

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class HcalHPDFilter : public edm::EDFilter {
   public:
      explicit HcalHPDFilter(const edm::ParameterSet&);
      ~HcalHPDFilter();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
      // ----------member data ---------------------------
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
HcalHPDFilter::HcalHPDFilter(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

}


HcalHPDFilter::~HcalHPDFilter()
{

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
HcalHPDFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel("hbhereco", hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());

   for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) {
     if (hhit->energy() > 5.) return true;
   }
   return false;
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalHPDFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalHPDFilter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalHPDFilter);

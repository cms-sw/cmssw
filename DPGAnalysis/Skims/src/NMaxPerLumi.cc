// -*- C++ -*-
//
// Package:    NMaxPerLumi
// Class:      NMaxPerLumi
// 
/**\class NMaxPerLumi NMaxPerLumi.cc WorkSpace/NMaxPerLumi/src/NMaxPerLumi.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Fri Apr  9 18:54:59 CEST 2010
// $Id: NMaxPerLumi.cc,v 1.4 2013/02/27 20:17:14 wmtan Exp $
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

//
// class declaration
//

class NMaxPerLumi : public edm::EDFilter {
   public:
      explicit NMaxPerLumi(const edm::ParameterSet&);
      ~NMaxPerLumi();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;
      
  // ----------member data ---------------------------
  std::map< unsigned int , std::map < unsigned int, unsigned int > > counters;
  unsigned int nMaxPerLumi_;
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
NMaxPerLumi::NMaxPerLumi(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  nMaxPerLumi_ = iConfig.getParameter<unsigned int>("nMaxPerLumi");
}


NMaxPerLumi::~NMaxPerLumi()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
NMaxPerLumi::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  const edm::EventID & id = iEvent.id();

  if (counters[id.run()][id.luminosityBlock()]>=nMaxPerLumi_)
    return false;
  else{
    counters[id.run()][id.luminosityBlock()]++;
    return true;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
NMaxPerLumi::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
NMaxPerLumi::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(NMaxPerLumi);

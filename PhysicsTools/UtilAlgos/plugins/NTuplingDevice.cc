// -*- C++ -*-
//
// Package:    NTuplingDevice
// Class:      NTuplingDevice
// 
/**\class NTuplingDevice NTuplingDevice.cc Workspace/NTuplingDevice/src/NTuplingDevice.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Sun May 11 21:12:46 CEST 2008
// $Id: NTuplingDevice.cc,v 1.4 2009/12/18 17:52:25 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/UtilAlgos/interface/NTupler.h"

//
// class decleration
//

class NTuplingDevice : public edm::EDProducer {
   public:
      explicit NTuplingDevice(const edm::ParameterSet&);
      ~NTuplingDevice();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
  NTupler * ntupler_;
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
NTuplingDevice::NTuplingDevice(const edm::ParameterSet& iConfig)
{
  //this Ntupler can work with the InputTagDistributor, but should not be configured as such.
  edm::ParameterSet ntPset = iConfig.getParameter<edm::ParameterSet>("Ntupler");
  std::string ntuplerName=ntPset.getParameter<std::string>("ComponentName");
  ntupler_ = NTuplerFactory::get()->create(ntuplerName, ntPset);

  //register the leaves from the ntupler
  ntupler_->registerleaves(this);

  //put a dummy product if the ntupler does not output on edm
  produces<double>("dummy");
}


NTuplingDevice::~NTuplingDevice(){}

// ------------ method called to produce the data  ------------
void
NTuplingDevice::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  ntupler_->fill(iEvent);
  std::auto_ptr<double> v(new double(0));
  iEvent.put(v,"dummy");
}

// ------------ method called once each job just before starting event loop  ------------
void 
NTuplingDevice::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
NTuplingDevice::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(NTuplingDevice);

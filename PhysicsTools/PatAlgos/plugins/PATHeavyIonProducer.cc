// -*- C++ -*-
//
// Package:    PATHeavyIonProducer
// Class:      PATHeavyIonProducer
// 
/**\class PATHeavyIonProducer PATHeavyIonProducer.cc yetkin/PATHeavyIonProducer/src/PATHeavyIonProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Aug 13 08:39:51 EDT 2009
//
//


// system include files
#include <memory>
#include <string>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HeavyIonEvent/interface/HeavyIon.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "HepMC/HeavyIon.h"

using namespace std;

//
// class decleration
//

class PATHeavyIonProducer : public edm::global::EDProducer<> {
public:
  explicit PATHeavyIonProducer(const edm::ParameterSet&);
  ~PATHeavyIonProducer();
  
private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  // ----------member data ---------------------------
  
  const bool doMC_;
  const bool doReco_;
  const std::vector<std::string> hepmcSrc_;
  const edm::InputTag centSrc_;
  const edm::InputTag evtPlaneSrc_;

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
PATHeavyIonProducer::PATHeavyIonProducer(const edm::ParameterSet& iConfig) :
  doMC_(iConfig.getParameter<bool>("doMC")),
  doReco_(iConfig.getParameter<bool>("doReco")),
  hepmcSrc_( doMC_ ? iConfig.getParameter<std::vector<std::string> >("generators") : std::vector<std::string>() ),
  centSrc_( doReco_ ? iConfig.getParameter<edm::InputTag>("centrality") : edm::InputTag() ),
  evtPlaneSrc_(doReco_ ? iConfig.getParameter<edm::InputTag>("evtPlane") : edm::InputTag() )
{
   //register your products
   produces<pat::HeavyIon>();   
}


PATHeavyIonProducer::~PATHeavyIonProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PATHeavyIonProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {

}

//define this as a plug-in
DEFINE_FWK_MODULE(PATHeavyIonProducer);

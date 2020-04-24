
/*
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/EDMException.h"

//#include "HepMC/GenEvent.h"
// #include "CLHEP/Vector/ThreeVector.h"
// #include "HepMC/SimpleVector.h"

using namespace edm;
using namespace CLHEP;
//using namespace HepMC;


BaseEvtVtxGenerator::BaseEvtVtxGenerator( const ParameterSet& pset ) 
{
   Service<RandomNumberGenerator> rng;
   if ( ! rng.isAvailable()) {
     throw cms::Exception("Configuration")
       << "The BaseEvtVtxGenerator requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file. \n" 
          "You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }

   sourceToken=consumes<edm::HepMCProduct>(pset.getParameter<edm::InputTag>("src"));
   produces<edm::HepMCProduct>();
}

BaseEvtVtxGenerator::~BaseEvtVtxGenerator() 
{
}

void BaseEvtVtxGenerator::produce( Event& evt, const EventSetup& )
{
   edm::Service<edm::RandomNumberGenerator> rng;
   CLHEP::HepRandomEngine* engine = &rng->getEngine(evt.streamID());

   Handle<HepMCProduct> HepUnsmearedMCEvt ;
   
   evt.getByToken( sourceToken, HepUnsmearedMCEvt ) ;
   
   // Copy the HepMC::GenEvent
   HepMC::GenEvent* genevt = new HepMC::GenEvent(*HepUnsmearedMCEvt->GetEvent());
   std::unique_ptr<edm::HepMCProduct> HepMCEvt(new edm::HepMCProduct(genevt));
   // generate new vertex & apply the shift 
   //
   HepMCEvt->applyVtxGen( newVertex(engine) ) ;

   //HepMCEvt->LorentzBoost( 0., 142.e-6 );
   HepMCEvt->boostToLab( GetInvLorentzBoost(), "vertex" );
   HepMCEvt->boostToLab( GetInvLorentzBoost(), "momentum" );
   
   evt.put(std::move(HepMCEvt)) ;
      
   return ;
}

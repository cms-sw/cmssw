
/*
*  $Date: 2009/05/25 12:46:03 $
*  $Revision: 1.14 $
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

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
using namespace std;
using namespace CLHEP;
//using namespace HepMC;

BaseEvtVtxGenerator::BaseEvtVtxGenerator( const ParameterSet& pset ) 
	: fVertex(0), boost_(0), fTimeOffset(0), fEngine(0),
	  sourceLabel(pset.getParameter<edm::InputTag>("src"))
{
   
/* No longer needed...

   // 1st of all, check on module_label - must be VtxSmeared !
   if ( pset.getParameter<string>("@module_label") != "VtxSmeared" )
   {
      throw cms::Exception("Configuration")
        << "Module has an invalid module label. "
           "The label of this module MUST be VtxSmeared.";
   }
*/
      
   Service<RandomNumberGenerator> rng;

   if ( ! rng.isAvailable()) {

     throw cms::Exception("Configuration")
       << "The BaseEvtVtxGenerator requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }

   CLHEP::HepRandomEngine& engine = rng->getEngine();
   fEngine = &engine;

      
   produces<bool>(); 
}

BaseEvtVtxGenerator::~BaseEvtVtxGenerator() 
{
   delete fVertex ;
   if (boost_ != 0 ) delete boost_;
   // no need since now it's done in HepMCProduct
   // delete fEvt ;
}

void BaseEvtVtxGenerator::produce( Event& evt, const EventSetup& )
{
   
   
   Handle<HepMCProduct> HepMCEvt ;
   
   evt.getByLabel( sourceLabel, HepMCEvt ) ;
   
   // generate new vertex & apply the shift 
   //
   HepMCEvt->applyVtxGen( newVertex() ) ;

   //HepMCEvt->LorentzBoost( 0., 142.e-6 );
   HepMCEvt->boostToLab( GetInvLorentzBoost(), "vertex" );
   HepMCEvt->boostToLab( GetInvLorentzBoost(), "momentum" );
   
   // OK, create a (pseudo)product and put in into edm::Event
   //
   auto_ptr<bool> NewProduct(new bool(true)) ;      
   evt.put( NewProduct ) ;
      
   return ;
}

CLHEP::HepRandomEngine& BaseEvtVtxGenerator::getEngine() 
{
   return *fEngine;
}

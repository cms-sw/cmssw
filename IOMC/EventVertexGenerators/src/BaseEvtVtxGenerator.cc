
/*
*  $Date: 2007/05/10 02:46:40 $
*  $Revision: 1.6 $
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/Exception.h"

//#include "HepMC/GenEvent.h"
// #include "CLHEP/Vector/ThreeVector.h"
// #include "HepMC/SimpleVector.h"

using namespace edm;
using namespace std;
using namespace CLHEP;
//using namespace HepMC;

BaseEvtVtxGenerator::BaseEvtVtxGenerator( const ParameterSet& pset ) 
	: boost_(0), fVertex(0), fEngine(0)
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

   HepRandomEngine& engine = rng->getEngine();
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
   evt.getByLabel( "source", HepMCEvt ) ;
            
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

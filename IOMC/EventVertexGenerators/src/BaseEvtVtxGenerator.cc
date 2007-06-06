
/*
*  $Date: 2006/09/29 17:02:11 $
*  $Revision: 1.1 $
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/Vector/ThreeVector.h"

using namespace edm;
using namespace std;
using namespace CLHEP;
using namespace HepMC;

BaseEvtVtxGenerator::BaseEvtVtxGenerator( const ParameterSet& pset ) 
  : fVertex(0), fEvt(0), fEngine(0)
{
   
   // 1st of all, check on module_label - must be VtxSmeared !
   if ( pset.getParameter<string>("@module_label") != "VtxSmeared" )
   {
      throw cms::Exception("Configuration")
        << "Module has an invalid module label. "
           "The label of this module MUST be VtxSmeared.";
   }
      
   Service<RandomNumberGenerator> rng;

   if ( ! rng.isAvailable()) {

     throw cms::Exception("Configuration")
       << "The BaseEvtVtxGenerator requires the RandomNumberGeneratorService\n"
          "which is not present in the configuration file.  You must add the service\n"
          "in the configuration file or remove the modules that require it.";
   }

   HepRandomEngine& engine = rng->getEngine();
   fEngine = &engine;

   produces<HepMCProduct>();   
}

BaseEvtVtxGenerator::~BaseEvtVtxGenerator() 
{
   delete fVertex ;
   // no need since now it's done in HepMCProduct
   // delete fEvt ;
}

void BaseEvtVtxGenerator::produce( Event& evt, const EventSetup& )
{
   
   vector< Handle<HepMCProduct> > AllHepMCEvt ;   
   evt.getManyByType( AllHepMCEvt ) ;
      
   for ( unsigned int i=0; i<AllHepMCEvt.size(); ++i )
   {
      if ( !AllHepMCEvt[i].isValid() )
      {
         // in principal, should never happen, as it's taken care of bt Framework
         throw cms::Exception("InvalidReference")
            << "Invalid reference to HepMCProduct\n";
      }
   
      // now the "real" check,
      // that is, whether there's or not HepMCProduct with VtxGen applied
      //
      // if there's already one, just bail out
      //
      if ( AllHepMCEvt[i].provenance()->moduleLabel() == "VtxSmeared" )
      {
         throw cms::Exception("LogicError")
            << "VtxSmeared HepMCProduce already exists\n";
      }
   }
   
   // Note : for some reason, creating an object (rather than a pointer)
   //        somehow creates rubish in the HepMCProduct, don't know why...
   //        so I've decided to go with a pointer
   //
   // no need for memory cleanup here - done in HepMCProduct
   //
   //if ( fEvt != NULL ) delete fEvt ;
   //
   fEvt = new GenEvent(*AllHepMCEvt[0]->GetEvent()) ;
         
   // vertex itself
   //
   Hep3Vector* VtxPos = newVertex() ;

   // here loop over NewEvent and shift with NewVtx
   //
   for ( GenEvent::vertex_iterator vt=fEvt->vertices_begin();
                                   vt!=fEvt->vertices_end(); ++vt )
   {
      double x = (*vt)->position().x() + VtxPos->x() ;
      double y = (*vt)->position().y() + VtxPos->y() ;
      double z = (*vt)->position().z() + VtxPos->z() ;
      (*vt)->set_position( HepLorentzVector(x,y,z) ) ;      
   }
         
   // OK, create a product and put in into edm::Event
   //
   auto_ptr<HepMCProduct> NewProduct(new HepMCProduct()) ;
   NewProduct->addHepMCData( fEvt ) ;
      
   evt.put( NewProduct ) ;
      
   return ;
}

CLHEP::HepRandomEngine& BaseEvtVtxGenerator::getEngine() 
{
   return *fEngine;
}

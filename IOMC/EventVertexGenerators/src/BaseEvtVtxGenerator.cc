
/*
*  $Date: $
*  $Revision: $
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

// #include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/Exception.h"


using namespace edm;
using namespace std;
using namespace CLHEP;
using namespace HepMC;

BaseEvtVtxGenerator::BaseEvtVtxGenerator( const ParameterSet& pset ) 
   : fEngine(0), fVertex(0), fEvt(0)
{
   
   // 1st of all, check on module_label - must be VtxSmeared !
   if ( pset.getParameter<string>("@module_label") != "VtxSmeared" )
   {
      throw Exception( errors::Configuration, 
                       "Invalid moduleLabel VG; the label of this module MUST be VtxSmeared" ) ;
   }
      
/*
   auto_ptr<EventVertexGeneratorMakerBase> 
      VtxGenMaker( EventVertexGeneratorFactory::get()->create
                   (pset.getParameter<std::string> ("type")) );
*/   
   Service<RandomNumberGenerator> rng;
   long seed = (long)(rng->mySeed()) ;
   // cout << " seed= " << seed << endl ;
   fEngine = new HepJamesRandom(seed) ;

/*
   if(VtxGenMaker.get()==0) 
   {
        throw Exception(errors::Configuration,
	                "Unable to find the event vertex generator requested") ;
   }
   fEventVertexGenerator = VtxGenMaker->make(pset,seed) ;

   if (fEventVertexGenerator.get()==0) 
      throw Exception(errors::Configuration,"EventVertexGenerator construction failed");
*/   
   produces<HepMCProduct>() ;
   
}

BaseEvtVtxGenerator::~BaseEvtVtxGenerator() 
{
   if ( fEngine != NULL ) delete fEngine ;
   if ( fVertex != NULL ) delete fVertex ;
   // no need since now it's done in HepMCProduct
   //if ( fEvt != NULL ) delete fEvt ;
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
	 throw Exception(errors::InvalidReference, "Invalid reference to HepMCProduct") ;
      }
   
      // now the "real" check,
      // that is, whether there's or not HepMCProduct with VtxGen applied
      //
      // if there's already one, just bail out
      //
      if ( AllHepMCEvt[i].provenance()->moduleLabel() == "VtxSmeared" )
      {
	 throw Exception(errors::LogicError, "VtxSmeared HepMCProduce already exists") ;
      }
   }
   
/*
   if ( fEventVertexGenerator.get() == 0 ) // overprotection
   {
      throw Exception(errors::Configuration, "Invalid VertexGenerator" ) ;
   }
*/   
   // Note : for some reason, creating an object (rather than a pointer)
   //        somehow creates rubish in the HepMCProduct, don't know why...
   //        so I've decided to go with a pointer
   //
   // no need for memory cleanup here - done in HepMCProduct
   //
   //if ( fEvt != NULL ) delete fEvt ;
   //
   fEvt = new GenEvent(*AllHepMCEvt[0]->GetEvent()) ;
         
   // vertex ietself
   //
   //Hep3Vector* VtxPos = fEventVertexGenerator.get()->newVertex() ; 
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

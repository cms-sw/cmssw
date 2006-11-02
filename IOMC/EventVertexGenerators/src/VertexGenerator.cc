
/*
*  $Date: 2006/08/08 23:00:18 $
*  $Revision: 1.6 $
*/

#include "IOMC/EventVertexGenerators/interface/VertexGenerator.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "IOMC/EventVertexGenerators/interface/EventVertexGeneratorFactory.h"

#include "FWCore/Utilities/interface/Exception.h"


using namespace edm;
using namespace std;
using namespace CLHEP;
using namespace HepMC;

VertexGenerator::VertexGenerator( const ParameterSet& pset ) 
   : fEvt(0)
{
   
   // 1st of all, check on module_label - must be VtxSmeared !
   if ( pset.getParameter<string>("@module_label") != "VtxSmeared" )
   {
      throw cms::Exception("Configuration")
        << "VertexGenerator has an invalid module label. The label of this module MUST be VtxSmeared.";
   }
      
   auto_ptr<EventVertexGeneratorMakerBase> 
      VtxGenMaker( EventVertexGeneratorFactory::get()->create
                   (pset.getParameter<std::string> ("type")) );
   
   if(VtxGenMaker.get()==0) 
   {
      throw cms::Exception("Configuration")
        << "Unable to find the type of vertex generator requested";
   }
   fEventVertexGenerator = VtxGenMaker->make(pset) ;

   if (fEventVertexGenerator.get()==0) 
      throw cms::Exception("Configuration")
        << "EventVertexGenerator construction failed";
   
   produces<HepMCProduct>() ;
}

VertexGenerator::~VertexGenerator() 
{
   // no need since now it's done in HepMCProduct
   //if ( fEvt != NULL ) delete fEvt ;
}

void VertexGenerator::produce( Event& evt, const EventSetup& )
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
   
   if ( fEventVertexGenerator.get() == 0 ) // overprotection
   {
      throw cms::Exception("Configuration")
         << "Invalid VertexGenerator\n";
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
         
   // vertex ietself
   //
   Hep3Vector* VtxPos = fEventVertexGenerator.get()->newVertex() ; 

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

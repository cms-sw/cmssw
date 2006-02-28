#ifndef IOMC_VertexGenerator_H
#define IOMC_VertexGenerator_H

/*
*   $Date: $
*   $Revision: $
*/

#include "FWCore/Framework/interface/EDProducer.h"

#include "CLHEP/HepMC/GenEvent.h"

// fwd declarations
//
class BaseEventVertexGenerator ;

namespace edm
{

   class VertexGenerator : public edm::EDProducer
   {

      public :
      
      // ctor & dtor
      //
      explicit VertexGenerator( const edm::ParameterSet& ) ;
      virtual ~VertexGenerator() ;
   
      virtual void produce( edm::Event&, const edm::EventSetup&);


      private :
   
      // data members
      //
      HepMC::GenEvent*                        fEvt;
      std::auto_ptr<BaseEventVertexGenerator> fEventVertexGenerator;
     
   } ;

}

#endif

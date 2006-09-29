#ifndef IOMC_BaseEvtVtxGenerator_H
#define IOMC_BaseEvtVtxGenerator_H
/*
*   $Date: $
*   $Revision: $
*/

#include "FWCore/Framework/interface/EDProducer.h"
 
#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Random/JamesRandom.h"

namespace edm
{

   class BaseEvtVtxGenerator : public edm::EDProducer
   {
      public :
      
      // ctor & dtor
      explicit BaseEvtVtxGenerator( const edm::ParameterSet& ) ;
      virtual ~BaseEvtVtxGenerator() ;
      
      virtual void produce( edm::Event&, const edm::EventSetup& );
      
      virtual Hep3Vector * newVertex() = 0;
  /** This methid - and the comment - is a left-over from COBRA-OSCAR time :
   *  return the last generated event vertex.
   *  If no vertex has been generated yet, a NULL pointer is returned. */
      virtual Hep3Vector * lastVertex() { return fVertex ; }
      
      protected:
      CLHEP::HepRandomEngine*  fEngine ;
      Hep3Vector*              fVertex ;
      
      private :
    
      // data members
      //
      HepMC::GenEvent*         fEvt;
   } ;

}

#endif

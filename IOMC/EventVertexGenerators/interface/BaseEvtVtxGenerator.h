#ifndef IOMC_BaseEvtVtxGenerator_H
#define IOMC_BaseEvtVtxGenerator_H
/*
*   $Date: 2006/11/07 19:38:31 $
*   $Revision: 1.2 $
*/

#include "FWCore/Framework/interface/EDProducer.h"

/*
namespace HepMC {
   class GenEvent;
}
*/

namespace CLHEP {
   class Hep3Vector;
   class HepRandomEngine;
}

class BaseEvtVtxGenerator : public edm::EDProducer
{
   public:
      
   // ctor & dtor
   explicit BaseEvtVtxGenerator( const edm::ParameterSet& );
   virtual ~BaseEvtVtxGenerator();
      
   virtual void produce( edm::Event&, const edm::EventSetup& );
      
   virtual CLHEP::Hep3Vector * newVertex() = 0;
   /** This method - and the comment - is a left-over from COBRA-OSCAR time :
    *  return the last generated event vertex.
    *  If no vertex has been generated yet, a NULL pointer is returned. */
   virtual CLHEP::Hep3Vector * lastVertex() { return fVertex; }

   protected:

   // Returns a reference to encourage users to use a reference
   // when initializing CLHEP distributions.  If a pointer
   // is used, then the distribution thinks it owns the engine
   // and will delete the engine when the distribution is destroyed
   // (a big problem since the distribution does not own the memory).
   CLHEP::HepRandomEngine& getEngine();

   CLHEP::Hep3Vector*       fVertex;

   private :

   CLHEP::HepRandomEngine*  fEngine;
};

#endif

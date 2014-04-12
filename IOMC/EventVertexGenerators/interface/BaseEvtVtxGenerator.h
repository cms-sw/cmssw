#ifndef IOMC_BaseEvtVtxGenerator_H
#define IOMC_BaseEvtVtxGenerator_H
/*
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TMatrixD.h"

/*
namespace HepMC {
   class GenEvent;
}
*/

namespace HepMC {
   class FourVector ;
}

namespace CLHEP {
   //class Hep3Vector;
   class HepRandomEngine;
}

class BaseEvtVtxGenerator : public edm::EDProducer
{
   public:
      
   // ctor & dtor
   explicit BaseEvtVtxGenerator( const edm::ParameterSet& );
   virtual ~BaseEvtVtxGenerator();
      
   virtual void produce( edm::Event&, const edm::EventSetup&) override;

   //virtual CLHEP::Hep3Vector* newVertex() = 0;
   virtual HepMC::FourVector* newVertex(CLHEP::HepRandomEngine*) = 0 ;
   /** This method - and the comment - is a left-over from COBRA-OSCAR time :
    *  return the last generated event vertex.
    *  If no vertex has been generated yet, a NULL pointer is returned. */
   //virtual CLHEP::Hep3Vector* lastVertex() { return fVertex; }
   virtual HepMC::FourVector* lastVertex() { return fVertex; }
   
   virtual TMatrixD* GetInvLorentzBoost() = 0;
   
   protected:

   //CLHEP::Hep3Vector*       fVertex;
   HepMC::FourVector*       fVertex ;
   TMatrixD *boost_;
   double fTimeOffset;
   
   private :

   edm::InputTag            sourceLabel;
   
};

#endif

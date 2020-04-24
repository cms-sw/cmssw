#ifndef IOMC_BaseEvtVtxGenerator_H
#define IOMC_BaseEvtVtxGenerator_H
/*
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "TMatrixD.h"

namespace HepMC {
   class FourVector ;
}

namespace CLHEP {
   class HepRandomEngine;
}

namespace edm {
   class HepMCProduct;
}

class BaseEvtVtxGenerator : public edm::stream::EDProducer<>
{
   public:
      
   // ctor & dtor
   explicit BaseEvtVtxGenerator( const edm::ParameterSet& );
   ~BaseEvtVtxGenerator() override;
      
   void produce( edm::Event&, const edm::EventSetup&) override;

   virtual HepMC::FourVector newVertex(CLHEP::HepRandomEngine*) const = 0 ;
   /** This method - and the comment - is a left-over from COBRA-OSCAR time :
    *  return the last generated event vertex.
    *  If no vertex has been generated yet, a NULL pointer is returned. */
   //virtual CLHEP::Hep3Vector* lastVertex() { return fVertex; }
   //virtual HepMC::FourVector* lastVertex() { return fVertex; }
   
   virtual TMatrixD const* GetInvLorentzBoost() const = 0;
      
   private :

   edm::EDGetTokenT<edm::HepMCProduct> sourceToken;
   
};

#endif

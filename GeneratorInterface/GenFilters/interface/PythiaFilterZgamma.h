#ifndef PythiaFilterZgamma_h
#define PythiaFilterZgamma_h

/** \class PythiaFilterZgamma
 *
 *  PythiaFilterZgamma filter implements generator-level preselections 
 *  for Z0 + photon like events to be used in TGC.
 * 
 * \author A.Kyriakis, NCSR "Demokritos" 
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class HepMCProduct;
}

class PythiaFilterZgamma : public edm::EDFilter {
   public:
      explicit PythiaFilterZgamma(const edm::ParameterSet&);
      ~PythiaFilterZgamma();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       
       int selProc; // sel_Proc = 1 : ->Z->e+e-, sel_Proc = 2: Z->mu+mu-
       
       double ptElMin;
       double ptMuMin;
       double ptPhotonMin;
       
       double etaElMax;
       double etaMuMax;
       double etaPhotonMax;

       int theNumberOfSelected;

};
#endif

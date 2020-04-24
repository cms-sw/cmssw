#ifndef PythiaFilterGammaJetIsoPi0_h
#define PythiaFilterGammaJetIsoPi0_h

/** \class PythiaFilterGammaJetIsoPi0
 *
 *  PythiaFilterGammaJetIsoPi0 filter implements generator-level preselections 
 *  for photon+jet like events to be used in pi0 rejection studies.
 *  Based on A. Ulyanov PythiaFilterGammaJetWithBg.h code 
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

class PythiaFilterGammaJetIsoPi0 : public edm::EDFilter {
   public:
      explicit PythiaFilterGammaJetIsoPi0(const edm::ParameterSet&);
      ~PythiaFilterGammaJetIsoPi0();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       double etaMin;
       double PtMin;
       double etaMax;
       double PtMax;        
       double isocone;
       double isodr;
       double ebEtaMax;
       double deltaEB;
       double deltaEE;

       int theNumberOfTestedEvt;
       int theNumberOfSelected;

};
#endif

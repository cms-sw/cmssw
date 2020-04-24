#ifndef PythiaFilterEMJet_h
#define PythiaFilterEMJet_h

/** \class PythiaFilterEMJet
 *
 *  PythiaFilterEMJet filter implements generator-level preselections 
 *
 * \author R.Salerno, Universita' Milano Bicocca
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

class PythiaFilterEMJet : public edm::EDFilter {
   public:
      explicit PythiaFilterEMJet(const edm::ParameterSet&);
      ~PythiaFilterEMJet();

      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       double etaMin;
       double eTSumMin;
       double pTMin;
       double etaMax;
       double eTSumMax;        
       double pTMax;        
       double ebEtaMax;
       double deltaEB;
       double deltaEE;

       int theNumberOfTestedEvt;
       int theNumberOfSelected;
       int maxnumberofeventsinrun;

};
#endif

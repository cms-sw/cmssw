// -*- C++ -*-
//
// Package:    EcalGlobalShowerContainmentCorrectionsVsEtaESProducer
// Class:      EcalGlobalShowerContainmentCorrectionsVsEtaESProducer
// 
/**\class EcalGlobalShowerContainmentCorrectionsVsEtaESProducer EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h User/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer/interface/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              global shower containment corrections as a function of eta

     
 \author  Paolo Meridiani
 \id $Id: $
*/

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalCorrections/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h"
#include "CondFormats/DataRecord/interface/EcalGlobalShowerContainmentCorrectionsVsEtaRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"



class EcalGlobalShowerContainmentCorrectionsVsEtaESProducer : public edm::ESProducer {

   public:
      EcalGlobalShowerContainmentCorrectionsVsEtaESProducer(const edm::ParameterSet&);
     ~EcalGlobalShowerContainmentCorrectionsVsEtaESProducer();

      typedef std::auto_ptr<EcalGlobalShowerContainmentCorrectionsVsEta> ReturnType;

      ReturnType produce(const EcalGlobalShowerContainmentCorrectionsVsEtaRcd&);
   private:
  

};


EcalGlobalShowerContainmentCorrectionsVsEtaESProducer::EcalGlobalShowerContainmentCorrectionsVsEtaESProducer(const edm::ParameterSet& iConfig)
{   
   setWhatProduced(this);
}


EcalGlobalShowerContainmentCorrectionsVsEtaESProducer::~EcalGlobalShowerContainmentCorrectionsVsEtaESProducer(){ }


//
// member functions
//

EcalGlobalShowerContainmentCorrectionsVsEtaESProducer::ReturnType
EcalGlobalShowerContainmentCorrectionsVsEtaESProducer::produce(const EcalGlobalShowerContainmentCorrectionsVsEtaRcd& iRecord)
{

   using namespace edm::es;
   using namespace std;

   auto_ptr<EcalGlobalShowerContainmentCorrectionsVsEta> pEcalGlobalShowerContainmentCorrectionsVsEta(new EcalGlobalShowerContainmentCorrectionsVsEta) ;
   
   double values[] = {   0.998959,       // 3x3 
			 0.00124547,	  
			 -0.000348259,
			 6.04065e-006,  
			 0.999032,       // 5x5 
			 7.90628e-005,
			 -0.000175699,
			 -2.60715e-007,
   };




     const size_t size = sizeof values / sizeof values[0];
     EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients coeff;
     std::copy(values,values+size,coeff.data);
     pEcalGlobalShowerContainmentCorrectionsVsEta->fillCorrectionCoefficients(coeff);

     return pEcalGlobalShowerContainmentCorrectionsVsEta ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalGlobalShowerContainmentCorrectionsVsEtaESProducer);

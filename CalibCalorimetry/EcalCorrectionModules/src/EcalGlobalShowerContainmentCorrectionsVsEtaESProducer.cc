// -*- C++ -*-
//
// Package:    EcalGlobalShowerContainmentCorrectionsVsEtaESProducer
// Class:      EcalGlobalShowerContainmentCorrectionsVsEtaESProducer
// 
/**\class EcalGlobalShowerContainmentCorrectionsVsEtaESProducer EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h User/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer/interface/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              global shower containment corrections as a function of eta

     
 \author  Paolo Meridiani
*/

// system include files
#include <memory>

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
     ~EcalGlobalShowerContainmentCorrectionsVsEtaESProducer() override;

      typedef std::unique_ptr<EcalGlobalShowerContainmentCorrectionsVsEta> ReturnType;

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

   auto pEcalGlobalShowerContainmentCorrectionsVsEta = std::make_unique<EcalGlobalShowerContainmentCorrectionsVsEta>();
   
   double values[] = {   43.77,       // 3x3 
			 1.,	  
			 -3.97e-006,
			 43.77,       // 5x5 
			 1.,	  
			 -3.97e-006,
   };
   
   const size_t size = sizeof values / sizeof values[0];
   EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients coeff;
   std::copy(values,values+size,coeff.data);
   pEcalGlobalShowerContainmentCorrectionsVsEta->fillCorrectionCoefficients(coeff);
   
   return pEcalGlobalShowerContainmentCorrectionsVsEta ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalGlobalShowerContainmentCorrectionsVsEtaESProducer);

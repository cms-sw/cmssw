// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "CalibCalorimetry/EcalCorrectionModules/interface/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h"


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

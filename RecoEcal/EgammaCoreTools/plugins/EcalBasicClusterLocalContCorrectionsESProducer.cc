// -*- C++ -*-

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include "FWCore/Framework/interface/ModuleFactory.h"
// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "CalibCalorimetry/EcalCorrectionModules/interface/EcalBasicClusterLocalContCorrectionsESProducer.h"
#include "RecoEcal/EgammaCoreTools/plugins/EcalBasicClusterLocalContCorrectionsESProducer.h"

EcalBasicClusterLocalContCorrectionsESProducer::EcalBasicClusterLocalContCorrectionsESProducer(const edm::ParameterSet& iConfig)
{   
   setWhatProduced(this);
}


EcalBasicClusterLocalContCorrectionsESProducer::~EcalBasicClusterLocalContCorrectionsESProducer(){ }


//
// member functions
//

EcalBasicClusterLocalContCorrectionsESProducer::ReturnType
EcalBasicClusterLocalContCorrectionsESProducer::produce(const EcalClusterLocalContCorrParametersRcd& iRecord)
{

   using namespace edm::es;
   using namespace std;

   auto_ptr<EcalClusterLocalContCorrParameters> pEcalClusterLocalContCorrParameters(new EcalClusterLocalContCorrParameters) ;

   double values[] = {  1.00603 , 0.00300789 , 0.0667232 , // local eta, mod1
			1.00655 , 0.00386189 , 0.073931  , // local eta, mod2
			1.00634 , 0.00631341 , 0.0764134 , // local eta, mod3
			1.00957 , 0.0113306 , 0.123808   , // local eta, mod4
			1.00402 , 0.00108324 , 0.0428149 , // local phi, mod1
			1.00393 , 0.000937121 , 0.041658 , // local phi, mod2
			1.00299 , 0.00126836 , 0.0321188 , // local phi, mod3
			1.00279 , -0.000700709 , 0.0293207 // local phi, mod4
   };

   size_t size = 24;
   pEcalClusterLocalContCorrParameters->params().resize(size);
   std::copy(values,values+size,pEcalClusterLocalContCorrParameters->params().begin());
   
   return pEcalClusterLocalContCorrParameters ;
}



DEFINE_FWK_EVENTSETUP_MODULE(EcalBasicClusterLocalContCorrectionsESProducer);

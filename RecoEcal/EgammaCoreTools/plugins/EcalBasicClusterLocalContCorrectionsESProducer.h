// -*- C++ -*-
//
// Package:    EcalBasicClusterLocalContCorrectionsESProducer
// Class:      EcalBasicClusterLocalContCorrectionsESProducer
// 
/**\class EcalBasicClusterLocalContCorrectionsESProducer EcalBasicClusterLocalContCorrectionsESProducer.h User/EcalBasicClusterLocalContCorrectionsESProducer/interface/EcalBasicClusterLocalContCorrectionsESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              shower containment corrections

     
 \author  Stefano Argiro
         Created:  Mon Mar  5 08:39:12 CET 2007
*/

#include "FWCore/Framework/interface/ESProducer.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/DataRecord/interface/EcalClusterLocalContCorrParametersRcd.h"

class EcalBasicClusterLocalContCorrectionsESProducer : public edm::ESProducer {

   public:
      EcalBasicClusterLocalContCorrectionsESProducer(const edm::ParameterSet&);
     ~EcalBasicClusterLocalContCorrectionsESProducer();

  typedef std::auto_ptr<EcalClusterLocalContCorrParameters> ReturnType;
  
  ReturnType produce(const EcalClusterLocalContCorrParametersRcd&);

   private:
  

};

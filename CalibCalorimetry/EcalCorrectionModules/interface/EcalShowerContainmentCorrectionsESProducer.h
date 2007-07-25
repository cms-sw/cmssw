// -*- C++ -*-
//
// Package:    EcalShowerContainmentCorrectionsESProducer
// Class:      EcalShowerContainmentCorrectionsESProducer
// 
/**\class EcalShowerContainmentCorrectionsESProducer EcalShowerContainmentCorrectionsESProducer.h User/EcalShowerContainmentCorrectionsESProducer/interface/EcalShowerContainmentCorrectionsESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              shower containment corrections

     
 \author  Stefano Argiro
         Created:  Mon Mar  5 08:39:12 CET 2007
 \id $Id: EcalShowerContainmentCorrectionsESProducer.cc,v 1.1 2007/05/15 20:46:31 argiro Exp $
*/

#include "FWCore/Framework/interface/ESProducer.h"
#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrections.h"
#include "CondFormats/DataRecord/interface/EcalShowerContainmentCorrectionsRcd.h"


class EcalShowerContainmentCorrectionsESProducer : public edm::ESProducer {

   public:
      EcalShowerContainmentCorrectionsESProducer(const edm::ParameterSet&);
     ~EcalShowerContainmentCorrectionsESProducer();

      typedef std::auto_ptr<EcalShowerContainmentCorrections> ReturnType;

      ReturnType produce(const EcalShowerContainmentCorrectionsRcd&);
   private:
  

};

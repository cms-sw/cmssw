// -*- C++ -*-
//
// Package:    EcalGlobalShowerContainmentCorrectionsVsEtaESProducer
// Class:      EcalGlobalShowerContainmentCorrectionsVsEtaESProducer
// 
/**\class EcalGlobalShowerContainmentCorrectionsVsEtaESProducer EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h User/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer/interface/EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              global shower containment corrections as a function of eta

     
 \author  Paolo Meridiani
 \id $Id: EcalGlobalShowerContainmentCorrectionsVsEtaESProducer.cc,v 1.2 2007/07/16 17:24:24 meridian Exp $
*/


#include "FWCore/Framework/interface/ESProducer.h"



#include "CondFormats/EcalCorrections/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h"
#include "CondFormats/DataRecord/interface/EcalGlobalShowerContainmentCorrectionsVsEtaRcd.h"




class EcalGlobalShowerContainmentCorrectionsVsEtaESProducer : public edm::ESProducer {

   public:
      EcalGlobalShowerContainmentCorrectionsVsEtaESProducer(const edm::ParameterSet&);
     ~EcalGlobalShowerContainmentCorrectionsVsEtaESProducer();

      typedef std::auto_ptr<EcalGlobalShowerContainmentCorrectionsVsEta> ReturnType;

      ReturnType produce(const EcalGlobalShowerContainmentCorrectionsVsEtaRcd&);
   private:
  

};

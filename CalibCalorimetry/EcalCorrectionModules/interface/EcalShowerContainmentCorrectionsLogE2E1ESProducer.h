// -*- C++ -*-
//
// Package:    EcalShowerContainmentCorrectionsLogE2E1ESProducer
// Class:      EcalShowerContainmentCorrectionsLogE2E1ESProducer
// 
/**\class EcalShowerContainmentCorrectionsLogE2E1ESProducer EcalShowerContainmentCorrectionsLogE2E1ESProducer.h User/EcalShowerContainmentCorrectionsLogE2E1ESProducer/interface/EcalShowerContainmentCorrectionsLogE2E1ESProducer.h

 Description: Trivial ESProducer to provide EventSetup with (hard coded)
              shower containment corrections

     
 \author  Stefano Argiro
         Created:  Mon Mar  5 08:39:12 CET 2007
 \id $Id: EcalShowerContainmentCorrectionsLogE2E1ESProducer.cc,v 1.1 2007/05/15 20:46:31 argiro Exp $
*/

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalCorrections/interface/EcalShowerContainmentCorrectionsLogE2E1.h"
#include "CondFormats/DataRecord/interface/EcalShowerContainmentCorrectionsLogE2E1Rcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"



class EcalShowerContainmentCorrectionsLogE2E1ESProducer : public edm::ESProducer {

   public:
      EcalShowerContainmentCorrectionsLogE2E1ESProducer(const edm::ParameterSet&);
     ~EcalShowerContainmentCorrectionsLogE2E1ESProducer();

      typedef std::auto_ptr<EcalShowerContainmentCorrectionsLogE2E1> ReturnType;

      ReturnType produce(const EcalShowerContainmentCorrectionsLogE2E1Rcd&);
   private:
  

};

// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTHcalIsolationProducers
// 
/**\class EgammaHLTHcalIsolationProducers EgammaHLTHcalIsolationProducers.cc RecoEgamma/EgammaHLTHcalIsolationProducers/interface/EgammaHLTHcalIsolationProducers.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Monica Vazquez Acosta
//         Created:  Tue Jun 13 14:48:33 CEST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class EgammaHLTHcalIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTHcalIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTHcalIsolationProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
};


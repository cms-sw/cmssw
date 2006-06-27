// -*- C++ -*-
//
// Package:    EgammaHLTProducers
// Class:      EgammaHLTEcalIsolationProducers
// 
/**\class EgammaHLTEcalIsolationProducers EgammaHLTEcalIsolationProducers.cc RecoEgamma/EgammaHLTEcalIsolationProducers/interface/EgammaHLTEcalIsolationProducers.h

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

class EgammaHLTEcalIsolationProducers : public edm::EDProducer {
   public:
      explicit EgammaHLTEcalIsolationProducers(const edm::ParameterSet&);
      ~EgammaHLTEcalIsolationProducers();


      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
};


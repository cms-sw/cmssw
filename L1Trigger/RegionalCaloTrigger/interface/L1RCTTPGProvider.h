// -*- C++ -*-
//
// Package:    L1RCTTPGProvider
// Class:      L1RCTTPGProvider
// 
/**\class L1RCTTPGProvider L1RCTTPGProvider.cc L1Trigger/L1RCTTPGProvider/src/L1RCTTPGProvider.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michail Bachtis
//         Created:  Tue Mar 10 18:29:22 CDT 2009
// $Id: L1RCTTPGProvider.h,v 1.2 2010/01/07 11:10:03 bachtis Exp $
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
// class decleration
//

class L1RCTTPGProvider : public edm::EDProducer {
   public:
      explicit L1RCTTPGProvider(const edm::ParameterSet&);
      ~L1RCTTPGProvider();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag ecalTPG_;
      edm::InputTag hcalTPG_;
      bool useHcalCosmicTiming;
      bool useEcalCosmicTiming;
      int preSamples;
      int postSamples;
      int hfShift;
      int hbShift;

};

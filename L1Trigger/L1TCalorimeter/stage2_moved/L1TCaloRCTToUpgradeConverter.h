#ifndef L1TCaloRCTToUpgradeConverter_h
#define L1TCaloRCTToUpgradeConverter_h

// -*- C++ -*-
//
// Package:    L1Trigger/skeleton
// Class:      skeleton
// 
/**\class skeleton skeleton.cc L1Trigger/skeleton/plugins/skeleton.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  James Brooke
//         Created:  Thu, 05 Dec 2013 17:39:27 GMT
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

namespace l1t {
    
  class L1TCaloRCTToUpgradeConverter : public edm::EDProducer { 
  public:
    explicit L1TCaloStage2Producer(const edm::ParameterSet& ps);
    ~L1TCaloStage2Producer();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions)
;
  private:
      virtual void beginJob() override;
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
      
      //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::Even
tSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventS
etup const&) override;

      // ----------member data ---------------------------

  }; 
  
} 

#endif

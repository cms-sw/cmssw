#ifndef L1TRawToDigi_h
#define L1TRawToDigi_h

// -*- C++ -*-
//
// Package:    EventFilter/L1TRawToDigi
// Class:      L1TDigiToRaw
// 
/**\class L1TDigiToRaw L1TDigiToRaw.cc EventFilter/L1TRawToDigi/plugins/L1TDigiToRaw.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Matthias Wolf
//         Created:  Mon, 10 Feb 2014 14:29:40 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class L1TDigiToRaw : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
      public:
         explicit L1TDigiToRaw(const edm::ParameterSet&);
         ~L1TDigiToRaw();

         static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

         using edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks>::consumes;

      private:
         virtual void beginJob() override;
         virtual void produce(edm::Event&, const edm::EventSetup&) override;
         virtual void endJob() override;

         virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {};
         virtual void endRun(edm::Run const&, edm::EventSetup const&) override {};
         virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {};
         virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {};

         // ----------member data ---------------------------
         // FIXME is actually fixed by the firmware version
         static const unsigned MAX_BLOCKS = 256;

         edm::InputTag inputLabel_;
         int fedId_;
         unsigned fwId_;

         PackerList packers_;
   };
}

#endif

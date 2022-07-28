// -*- C++ -*-
//
// Package:    L1RCTTPGProvider
// Class:      L1RCTTPGProvider
//
/**\class L1RCTTPGProvider L1RCTTPGProvider.cc
 L1Trigger/L1RCTTPGProvider/src/L1RCTTPGProvider.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michail Bachtis
//         Created:  Tue Mar 10 18:29:22 CDT 2009
//
//

// system include files
#include <memory>
#include <atomic>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//
// class decleration
//

class L1RCTTPGProvider : public edm::global::EDProducer<> {
public:
  explicit L1RCTTPGProvider(const edm::ParameterSet &);
  ~L1RCTTPGProvider() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalTPG_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalTPG_;
  mutable std::atomic<bool> useHcalCosmicTiming;
  mutable std::atomic<bool> useEcalCosmicTiming;
  int preSamples;
  int postSamples;
  int hfShift;
  int hbShift;
};

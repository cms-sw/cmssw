#ifndef MaskedRctInputDigiProducer_h
#define MaskedRctInputDigiProducer_h

// -*- C++ -*-
//
// Package:    MaskedRctInputDigiProducer
// Class:      MaskedRctInputDigiProducer
//
/**\class MaskedRctInputDigiProducer MaskedRctInputDigiProducer.cc
L1Trigger/MaskedRctInputDigiProducer/src/MaskedRctInputDigiProducer.cc

 Description: Takes collections of ECAL and HCAL digis, masks some towers
according to a mask in a text file, and creates new collections for use by the
RCT.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/65
//         Created:  Fri Nov 23 12:08:31 CET 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"

//
// class declaration
//

class MaskedRctInputDigiProducer : public edm::global::EDProducer<> {
public:
  explicit MaskedRctInputDigiProducer(const edm::ParameterSet &);
  ~MaskedRctInputDigiProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // ----------member data ---------------------------

  bool useEcal_;
  bool useHcal_;
  edm::EDGetTokenT<EcalTrigPrimDigiCollection> ecalDigisToken_;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> hcalDigisToken_;
  edm::FileInPath maskFile_;
};
#endif

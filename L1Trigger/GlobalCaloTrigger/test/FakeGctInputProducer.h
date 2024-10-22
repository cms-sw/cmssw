// -*- C++ -*-
//
// Package:    L1FakeRctProducer
// Class:      FakeGctInputProducer
//
/**\class FakeGctInputProducer FakeGctInputProducer.h L1Trigger/GlobalCaloTrigger/src/FakeGctInputProducer.h

 \brief EDProducer to fill GCT input buffers for testing

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu Nov 16 00:07:32 CET 2006
//
//

#ifndef FAKEGCTINPUTPRODUCER_H
#define FAKEGCTINPUTPRODUCER_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class FakeGctInputProducer : public edm::global::EDProducer<> {
public:
  explicit FakeGctInputProducer(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  int rgnMode_;
  int iemMode_;
  int niemMode_;
};

#endif

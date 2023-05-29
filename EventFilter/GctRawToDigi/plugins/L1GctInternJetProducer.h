#ifndef L1ExtraFromDigis_L1GctInternJetProducer_h
#define L1ExtraFromDigis_L1GctInternJetProducer_h
// -*- C++ -*-
//
// Package:     EventFilter/GctRawToDigi
// Class  :     L1GctInternJetProducer
//
/**\class L1GctInternJetProducer \file L1GctInternJetProducer.h EventFilter/GctRawToDigi/plugins/L1GctInternJetProducer.h 

\author Alex Tapper

 Description: producer of L1Extra style internal GCT jets from Level-1 hardware objects.

*/

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"

// forward declarations
class L1CaloGeometry;

class L1GctInternJetProducer : public edm::global::EDProducer<> {
public:
  explicit L1GctInternJetProducer(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::InputTag internalJetSource_;
  edm::ESGetToken<L1CaloGeometry, L1CaloGeometryRecord> caloGeomToken_;
  edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> jetScaleToken_;
  bool centralBxOnly_;
};

#endif

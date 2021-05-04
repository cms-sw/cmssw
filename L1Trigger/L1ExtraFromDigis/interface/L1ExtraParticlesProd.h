#ifndef L1ExtraFromDigis_L1ExtraParticlesProd_h
#define L1ExtraFromDigis_L1ExtraParticlesProd_h
// -*- C++ -*-
//
// Package:     L1ExtraFromDigis
// Class  :     L1ExtraParticlesProd
//
/**\class L1ExtraParticlesProd \file L1ExtraParticlesProd.h
 L1Trigger/L1ExtraFromDigis/interface/L1ExtraParticlesProd.h \author Werner Sun

 Description: producer of L1Extra particle objects from Level-1 hardware
 objects.

*/
//
// Original Author:
//         Created:  Tue Oct 17 00:13:51 EDT 2006
//

// system include files

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
// forward declarations
class L1CaloGeometry;

class L1ExtraParticlesProd : public edm::stream::EDProducer<> {
public:
  explicit L1ExtraParticlesProd(const edm::ParameterSet &);
  ~L1ExtraParticlesProd() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  //      math::XYZTLorentzVector gctLorentzVector( const double& et,
  math::PtEtaPhiMLorentzVector gctLorentzVector(const double &et,
                                                const L1GctCand &cand,
                                                const L1CaloGeometry *geom,
                                                bool central);

  // ----------member data ---------------------------
  bool produceMuonParticles_;
  edm::InputTag muonSource_;

  bool produceCaloParticles_;
  edm::InputTag isoEmSource_;
  edm::InputTag nonIsoEmSource_;
  edm::InputTag cenJetSource_;
  edm::InputTag forJetSource_;
  edm::InputTag tauJetSource_;
  edm::InputTag isoTauJetSource_;
  edm::InputTag etTotSource_;
  edm::InputTag etHadSource_;
  edm::InputTag etMissSource_;
  edm::InputTag htMissSource_;
  edm::InputTag hfRingEtSumsSource_;
  edm::InputTag hfRingBitCountsSource_;

  static const double muonMassGeV_;

  bool centralBxOnly_;

  // Set this to true when rerunning on RAW data where the GCT did not
  // produce a L1GctHtMiss record.
  bool ignoreHtMiss_;
  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> muScalesToken_;
  edm::ESGetToken<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd> muPtScaleToken_;
  edm::ESGetToken<L1CaloGeometry, L1CaloGeometryRecord> caloGeomToken_;
  edm::ESGetToken<L1CaloEtScale, L1EmEtScaleRcd> emScaleToken_;
  edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> jetScaleToken_;
  edm::ESGetToken<L1GctJetFinderParams, L1GctJetFinderParamsRcd> jetFinderParamsToken_;
  edm::ESGetToken<L1CaloEtScale, L1HtMissScaleRcd> htMissScaleToken_;
  edm::ESGetToken<L1CaloEtScale, L1HfRingEtScaleRcd> hfRingEtScaleToken_;
};

#endif

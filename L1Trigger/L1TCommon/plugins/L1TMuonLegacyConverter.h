#ifndef L1TCommon_L1TMuonLegacyConverter_h
#define L1TCommon_L1TMuonLegacyConverter_h
// -*- C++ -*-
//
// Package:     L1TCommon
// Class  :     L1TMuonLegacyConverter
//
/**\class L1TMuonLegacyConverter \file L1TMuonLegacyConverter.h L1Trigger/L1TCommon/interface/L1TMuonLegacyConverter.h

 Description: conver L1T muons legacy format to stage2 format

*/
//
// Original Author:  Pierluigi Bortignon
//         Created:  Sun March 6 2016
//

// system include files

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// #include "DataFormats/L1Trigger/interface/L1EmParticle.h"
// #include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
// #include "DataFormats/L1Trigger/interface/L1JetParticle.h"
// #include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
// #include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
// #include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
// #include "DataFormats/L1Trigger/interface/L1HFRings.h"
// #include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

//include new muon data format
#include "DataFormats/L1Trigger/interface/Muon.h"

// forward declarations
class L1CaloGeometry;
class L1MuTriggerScales;
class L1MuTriggerScalesRcd;
class L1MuTriggerPtScale;
class L1MuTriggerPtScaleRcd;

class L1TMuonLegacyConverter : public edm::stream::EDProducer<> {
public:
  explicit L1TMuonLegacyConverter(const edm::ParameterSet&);
  ~L1TMuonLegacyConverter() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // //      math::XYZTLorentzVector gctLorentzVector( const double& et,
  // math::PtEtaPhiMLorentzVector gctLorentzVector( const double& et,
  //      const L1GctCand& cand,
  //      const L1CaloGeometry* geom,
  //      bool central ) ;

  // ----------member data ---------------------------
  bool produceMuonParticles_;
  edm::InputTag muonSource_InputTag;
  edm::EDGetTokenT<L1MuGMTReadoutCollection> muonSource_InputToken;

  edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> muScalesToken_;
  edm::ESGetToken<L1MuTriggerPtScale, L1MuTriggerPtScaleRcd> muPtScaleToken_;
  // bool produceCaloParticles_ ;
  // edm::InputTag isoEmSource_ ;
  // edm::InputTag nonIsoEmSource_ ;
  // edm::InputTag cenJetSource_ ;
  // edm::InputTag forJetSource_ ;
  // edm::InputTag tauJetSource_ ;
  // edm::InputTag isoTauJetSource_ ;
  // edm::InputTag etTotSource_ ;
  // edm::InputTag etHadSource_ ;
  // edm::InputTag etMissSource_ ;
  // edm::InputTag htMissSource_ ;
  // edm::InputTag hfRingEtSumsSource_ ;
  // edm::InputTag hfRingBitCountsSource_ ;

  static const double muonMassGeV_;

  bool centralBxOnly_;

  // Set this to true when rerunning on RAW data where the GCT did not
  // produce a L1GctHtMiss record.
  bool ignoreHtMiss_;
};

#endif

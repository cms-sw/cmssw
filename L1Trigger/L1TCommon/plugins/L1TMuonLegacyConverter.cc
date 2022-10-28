// -*- C++ -*-
//
// Package:    L1TMuonLegacyConverter
// Class:      L1TMuonLegacyConverter
//
/**\class L1TMuonLegacyConverter \file L1TMuonLegacyConverter.cc src/L1TMuonLegacyConverter/src/L1TMuonLegacyConverter.cc
*/
//
// Original Author:  Bortignon Pierluigi
//         Created:  Sun March 6 EDT 2016
//
//

// system include files
#include <memory>

// user include files
#include "L1Trigger/L1TCommon/plugins/L1TMuonLegacyConverter.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// #include "FWCore/Utilities/interface/EDMException.h"

//
// class decleration
//

//
// constants, enums and typedefs
//
using namespace l1t;

//
// static data member definitions
//

double const L1TMuonLegacyConverter::muonMassGeV_ = 0.105658369;  // PDG06

//
// constructors and destructor
//
L1TMuonLegacyConverter::L1TMuonLegacyConverter(const edm::ParameterSet& iConfig) {
  using namespace l1extra;

  // moving inputTag here
  muonSource_InputTag = iConfig.getParameter<edm::InputTag>("muonSource");
  produceMuonParticles_ = iConfig.getUntrackedParameter<bool>("produceMuonParticles");
  centralBxOnly_ = iConfig.getUntrackedParameter<bool>("centralBxOnly");

  produces<MuonBxCollection>("imdMuonsLegacy");
  muonSource_InputToken = consumes<L1MuGMTReadoutCollection>(muonSource_InputTag);

  if (produceMuonParticles_) {
    muScalesToken_ = esConsumes();
    muPtScaleToken_ = esConsumes();
  }
}

L1TMuonLegacyConverter::~L1TMuonLegacyConverter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void L1TMuonLegacyConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace l1extra;
  using namespace std;

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~~~~~~~ Muons ~~~~~~~~~~~~~~~~~~~~
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  std::unique_ptr<MuonBxCollection> imdMuonsLegacy(new MuonBxCollection());

  LogDebug("L1TMuonLegacyConverter") << "\nWarning: L1MuGMTReadoutCollection with " << muonSource_InputTag
                                     << "\nrequested in configuration, but not found in the event." << std::endl;

  if (produceMuonParticles_) {
    ESHandle<L1MuTriggerScales> muScales = iSetup.getHandle(muScalesToken_);

    ESHandle<L1MuTriggerPtScale> muPtScale = iSetup.getHandle(muPtScaleToken_);

    Handle<L1MuGMTReadoutCollection> simMuCollection;
    iEvent.getByToken(muonSource_InputToken, simMuCollection);

    vector<L1MuGMTExtendedCand> simMuCands;

    if (!simMuCollection.isValid()) {
      LogDebug("L1TMuonLegacyConverter") << "\nWarning: L1MuGMTReadoutCollection with " << muonSource_InputTag
                                         << "\nrequested in configuration, but not found in the event." << std::endl;
    } else {
      if (centralBxOnly_) {
        // Get GMT candidates from central bunch crossing only
        simMuCands = simMuCollection->getRecord().getGMTCands();
      } else {
        // Get GMT candidates from all bunch crossings
        vector<L1MuGMTReadoutRecord> records = simMuCollection->getRecords();
        vector<L1MuGMTReadoutRecord>::const_iterator rItr = records.begin();
        vector<L1MuGMTReadoutRecord>::const_iterator rEnd = records.end();

        for (; rItr != rEnd; ++rItr) {
          vector<L1MuGMTExtendedCand> tmpCands = rItr->getGMTCands();

          simMuCands.insert(simMuCands.end(), tmpCands.begin(), tmpCands.end());
        }
      }

      vector<L1MuGMTExtendedCand>::const_iterator muItr = simMuCands.begin();
      vector<L1MuGMTExtendedCand>::const_iterator muEnd = simMuCands.end();
      for (int i = 0; muItr != muEnd; ++muItr, ++i) {
        if (!muItr->empty()) {
          // keep x and y components non-zero and protect against roundoff.
          double pt = muPtScale->getPtScale()->getLowEdge(muItr->ptIndex()) + 1.e-6;
          std::cout << "pt from muPtScale = " << pt << std::endl;
          double eta = muScales->getGMTEtaScale()->getCenter(muItr->etaIndex());
          double phi = muScales->getPhiScale()->getLowEdge(muItr->phiIndex());

          math::PtEtaPhiMLorentzVector p4(pt, eta, phi, muonMassGeV_);

          Muon outMu{p4,
                     (int)0,
                     (int)0,
                     (int)0,
                     (int)muItr->quality(),
                     (int)muItr->charge(),
                     (int)muItr->charge_valid(),
                     (int)muItr->isol(),
                     (int)0,
                     (int)0,
                     true,
                     (int)0,
                     (int)0,
                     (int)0,
                     (int)muItr->rank()};
          imdMuonsLegacy->push_back(muItr->bx(), outMu);
        }
      }
    }
  }

  iEvent.put(std::move(imdMuonsLegacy), "imdMuonsLegacy");

}  // closing produce

//define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonLegacyConverter);

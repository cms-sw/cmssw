// -*- C++ -*-
//
// Package:    L1Trigger/Phase2L1GMT
// Class:      Phase2L1TGMTSAMuonProducer
//
/**\class Phase2L1TGMTSAMuonProducer Phase2L1TGMTSAMuonProducer.cc L1Trigger/Phase2L1GMT/plugins/Phase2L1TGMTSAMuonProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Zhenbin Wu
//         Created:  Fri, 30 Apr 2021 19:10:59 GMT
//
//

#ifndef PHASE2GMT_SAMUONPRODUCER
#define PHASE2GMT_SAMUONPRODUCER

// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

#include "DataFormats/L1TMuonPhase2/interface/Constants.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTSAMuonProducer : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTSAMuonProducer(const edm::ParameterSet&);
  ~Phase2L1TGMTSAMuonProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  l1t::SAMuon Convertl1tMuon(const l1t::Muon& mu, const int bx_);

  // ----------member data ---------------------------
  edm::EDGetTokenT<BXVector<l1t::Muon> > muonToken_;
  unsigned int Nprompt;
  unsigned int Ndisplaced;
};

Phase2L1TGMTSAMuonProducer::Phase2L1TGMTSAMuonProducer(const edm::ParameterSet& iConfig)
    : muonToken_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("muonToken"))),
      Nprompt(iConfig.getParameter<uint>("Nprompt")),
      Ndisplaced(iConfig.getParameter<uint>("Ndisplaced")) {
  produces<std::vector<l1t::SAMuon> >("promptSAMuons").setBranchAlias("prompt");
  produces<std::vector<l1t::SAMuon> >("displacedSAMuons").setBranchAlias("displaced");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1TGMTSAMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::Handle<l1t::MuonBxCollection> muon;
  iEvent.getByToken(muonToken_, muon);

  // Output
  std::vector<SAMuon> prompt;
  std::vector<SAMuon> displaced;

  for (int bx = muon->getFirstBX(); bx <= muon->getLastBX(); ++bx) {
    //TODO: We are expecting to send all BX. Using bx0 for now
    if (bx != 0) {
      continue;
    }

    for (uint i = 0; i < muon->size(bx); ++i) {
      const l1t::Muon& mu = muon->at(bx, i);

      //TODO: Still looking for a way to get displaced muon
      if (abs(mu.hwDXY()) > 0)
        displaced.push_back(Convertl1tMuon(mu, bx));
      else
        prompt.push_back(Convertl1tMuon(mu, bx));
    }

    // Sort by hwPt
    std::sort(prompt.begin(), prompt.end(), std::greater<>());
    std::sort(displaced.begin(), displaced.end(), std::greater<>());

    // Store into output, allow up to 18 prompt + 18 displayed
    if (prompt.size() > Nprompt) {
      prompt.resize(Nprompt);
    }
    if (displaced.size() > Ndisplaced) {
      displaced.resize(Ndisplaced);
    }
  }

  std::unique_ptr<std::vector<l1t::SAMuon> > prompt_ptr = std::make_unique<std::vector<l1t::SAMuon> >(prompt);
  std::unique_ptr<std::vector<l1t::SAMuon> > displaced_ptr = std::make_unique<std::vector<l1t::SAMuon> >(displaced);
  iEvent.put(std::move(prompt_ptr), "promptSAMuons");
  iEvent.put(std::move(displaced_ptr), "displacedSAMuons");
}

// ===  FUNCTION  ============================================================
//         Name:  Phase2L1TGMTSAMuonProducer::Convertl1tMuon
//  Description:
// ===========================================================================
SAMuon Phase2L1TGMTSAMuonProducer::Convertl1tMuon(const l1t::Muon& mu, const int bx_) {
  qual_sa_t qual = mu.hwQual();
  int charge = mu.charge() > 0 ? 0 : 1;

  pt_sa_t pt = round(mu.pt() / LSBpt);
  phi_sa_t phi = round(mu.phi() / LSBphi);
  eta_sa_t eta = round(mu.eta() / LSBeta);
  // FIXME: Below are not well defined in phase1 GMT
  // Using the version from Correlator for now
  z0_sa_t z0 = 0;  // No tracks info in Phase 1
  // Use 2 bits with LSB = 30cm for BMTF and 25cm for EMTF currently, but subjet to change
  d0_sa_t d0 = mu.hwDXY();

  int bstart = 0;
  wordtype word(0);
  bstart = wordconcat<wordtype>(word, bstart, pt > 0, 1);
  bstart = wordconcat<wordtype>(word, bstart, pt, BITSGTPT);
  bstart = wordconcat<wordtype>(word, bstart, phi, BITSGTPHI);
  bstart = wordconcat<wordtype>(word, bstart, eta, BITSGTETA);
  bstart = wordconcat<wordtype>(word, bstart, z0, BITSSAZ0);
  bstart = wordconcat<wordtype>(word, bstart, d0, BITSSAD0);
  bstart = wordconcat<wordtype>(word, bstart, charge, 1);
  bstart = wordconcat<wordtype>(word, bstart, qual, BITSSAQUAL);

  SAMuon samuon(mu, charge, pt.to_uint(), eta.to_int(), phi.to_int(), z0.to_int(), d0.to_int(), qual.to_uint());
  samuon.setWord(word);
  return samuon;
}  // -----  end of function Phase2L1TGMTSAMuonProducer::Convertl1tMuon  -----

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase2L1TGMTSAMuonProducer::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase2L1TGMTSAMuonProducer::endStream() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1TGMTSAMuonProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("muonToken", edm::InputTag("simGmtStage2Digis"));
  desc.add<unsigned int>("Nprompt", 12);
  desc.add<unsigned int>("Ndisplaced", 12);
  descriptions.add("standaloneMuons", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTSAMuonProducer);
#endif

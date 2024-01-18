// -*- C++ -*-
#ifndef PHASE2GMT_SAMUONGHOSTCLEANER
#define PHASE2GMT_SAMUONGHOSTCLEANER

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
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "L1Trigger/Phase2L1GMT/interface/SAMuonCleaner.h"
//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTSAMuonGhostCleaner : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTSAMuonGhostCleaner(const edm::ParameterSet&);
  ~Phase2L1TGMTSAMuonGhostCleaner() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > barrelTokenPrompt_;
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > barrelTokenDisp_;
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > fwdToken_;

  SAMuonCleaner ghostCleaner;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
Phase2L1TGMTSAMuonGhostCleaner::Phase2L1TGMTSAMuonGhostCleaner(const edm::ParameterSet& iConfig)
    : barrelTokenPrompt_(consumes<std::vector<l1t::SAMuon> >(iConfig.getParameter<edm::InputTag>("barrelPrompt"))),
      barrelTokenDisp_(consumes<std::vector<l1t::SAMuon> >(iConfig.getParameter<edm::InputTag>("barrelDisp"))),
      fwdToken_(consumes<std::vector<l1t::SAMuon> >(iConfig.getParameter<edm::InputTag>("forward"))) {
  produces<std::vector<l1t::SAMuon> >("prompt");
  produces<std::vector<l1t::SAMuon> >("displaced");
}

Phase2L1TGMTSAMuonGhostCleaner::~Phase2L1TGMTSAMuonGhostCleaner() {}

// ------------ method called to produce the data  ------------
void Phase2L1TGMTSAMuonGhostCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<std::vector<l1t::SAMuon> > barrelPrompt;
  iEvent.getByToken(barrelTokenPrompt_, barrelPrompt);

  edm::Handle<std::vector<l1t::SAMuon> > barrelDisp;
  iEvent.getByToken(barrelTokenDisp_, barrelDisp);

  edm::Handle<std::vector<l1t::SAMuon> > forward;
  iEvent.getByToken(fwdToken_, forward);

  std::vector<l1t::SAMuon> muons = *barrelPrompt.product();
  muons.insert(muons.end(), forward->begin(), forward->end());

  std::vector<l1t::SAMuon> cleanedMuons = ghostCleaner.cleanTFMuons(muons);
  //here switch to the offical word required by the GT
  std::vector<l1t::SAMuon> finalPrompt;
  for (const auto& mu : cleanedMuons) {
    l1t::SAMuon m = mu;
    if (m.tfType() == l1t::tftype::bmtf)
      m.setHwQual(m.hwQual() >> 4);
    int bstart = 0;
    wordtype word(0);
    bstart = wordconcat<wordtype>(word, bstart, 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPt(), BITSGTPT);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPhi(), BITSGTPHI);
    bstart = wordconcat<wordtype>(word, bstart, m.hwEta(), BITSGTETA);
    bstart = wordconcat<wordtype>(word, bstart, 0, BITSSAZ0);
    bstart = wordconcat<wordtype>(word, bstart, m.hwD0(), BITSSAD0);
    bstart = wordconcat<wordtype>(word, bstart, m.charge() > 0 ? 0 : 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, m.hwQual(), BITSSAQUAL);
    m.setWord(word);
    finalPrompt.push_back(m);
  }

  std::vector<l1t::SAMuon> finalDisp;
  for (const auto& mu : *barrelDisp.product()) {
    l1t::SAMuon m = mu;
    if (m.tfType() == l1t::tftype::bmtf)
      m.setHwQual(m.hwQual() >> 4);
    int bstart = 0;
    wordtype word(0);
    bstart = wordconcat<wordtype>(word, bstart, 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPt(), BITSGTPT);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPhi(), BITSGTPHI);
    bstart = wordconcat<wordtype>(word, bstart, m.hwEta(), BITSGTETA);
    bstart = wordconcat<wordtype>(word, bstart, 0, BITSSAZ0);
    bstart = wordconcat<wordtype>(word, bstart, m.hwD0(), BITSSAD0);
    bstart = wordconcat<wordtype>(word, bstart, m.charge() > 0 ? 0 : 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, m.hwQual(), BITSSAQUAL);
    m.setWord(word);
    finalDisp.push_back(m);
  }

  std::unique_ptr<std::vector<l1t::SAMuon> > prompt_ptr = std::make_unique<std::vector<l1t::SAMuon> >(finalPrompt);
  std::unique_ptr<std::vector<l1t::SAMuon> > disp_ptr = std::make_unique<std::vector<l1t::SAMuon> >(finalDisp);
  iEvent.put(std::move(prompt_ptr), "prompt");
  iEvent.put(std::move(disp_ptr), "displaced");
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void Phase2L1TGMTSAMuonGhostCleaner::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void Phase2L1TGMTSAMuonGhostCleaner::endStream() {
  // please remove this method if not needed
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1TGMTSAMuonGhostCleaner::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(Phase2L1TGMTSAMuonGhostCleaner);

#endif

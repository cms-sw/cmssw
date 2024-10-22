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
  ~Phase2L1TGMTSAMuonGhostCleaner() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::vector<l1t::SAMuon> prodMuons(std::vector<l1t::SAMuon>& muons);

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > barrelTokenPrompt_;
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > barrelTokenDisp_;
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > fwdTokenPrompt_;
  edm::EDGetTokenT<std::vector<l1t::SAMuon> > fwdTokenDisp_;

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
      fwdTokenPrompt_(consumes<std::vector<l1t::SAMuon> >(iConfig.getParameter<edm::InputTag>("forwardPrompt"))),
      fwdTokenDisp_(consumes<std::vector<l1t::SAMuon> >(iConfig.getParameter<edm::InputTag>("forwardDisp"))) {
  produces<std::vector<l1t::SAMuon> >("prompt");
  produces<std::vector<l1t::SAMuon> >("displaced");
}

// ===  FUNCTION  ============================================================
//         Name:  Phase2L1TGMTSAMuonGhostCleaner::prodMuons
//  Description:
// ===========================================================================
std::vector<l1t::SAMuon> Phase2L1TGMTSAMuonGhostCleaner::prodMuons(std::vector<l1t::SAMuon>& muons) {
  std::vector<l1t::SAMuon> cleanedMuons = ghostCleaner.cleanTFMuons(muons);
  //here switch to the offical word required by the GT
  std::vector<l1t::SAMuon> finalMuons;
  for (const auto& mu : cleanedMuons) {
    l1t::SAMuon m = mu;
    if (m.tfType() == l1t::tftype::bmtf)
      m.setHwQual(m.hwQual() >> 4);
    int bstart = 0;
    wordtype word(0);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPt() > 0, 1);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPt(), BITSGTPT);
    bstart = wordconcat<wordtype>(word, bstart, m.hwPhi(), BITSGTPHI);
    bstart = wordconcat<wordtype>(word, bstart, m.hwEta(), BITSGTETA);
    bstart = wordconcat<wordtype>(word, bstart, m.hwZ0(), BITSSAZ0);
    bstart = wordconcat<wordtype>(word, bstart, m.hwD0(), BITSSAD0);
    bstart = wordconcat<wordtype>(word, bstart, m.hwCharge(), 1);
    wordconcat<wordtype>(word, bstart, m.hwQual(), BITSSAQUAL);
    m.setWord(word);
    finalMuons.push_back(m);
  }
  return finalMuons;
}  // -----  end of function Phase2L1TGMTSAMuonGhostCleaner::prodMuons  -----

// ------------ method called to produce the data  ------------
void Phase2L1TGMTSAMuonGhostCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<std::vector<l1t::SAMuon> > barrelPrompt;
  iEvent.getByToken(barrelTokenPrompt_, barrelPrompt);

  edm::Handle<std::vector<l1t::SAMuon> > barrelDisp;
  iEvent.getByToken(barrelTokenDisp_, barrelDisp);

  edm::Handle<std::vector<l1t::SAMuon> > forwardPrompt;
  iEvent.getByToken(fwdTokenPrompt_, forwardPrompt);

  edm::Handle<std::vector<l1t::SAMuon> > forwardDisp;
  iEvent.getByToken(fwdTokenDisp_, forwardDisp);

  // Prompt muons
  std::vector<l1t::SAMuon> muons = *barrelPrompt.product();
  muons.insert(muons.end(), forwardPrompt->begin(), forwardPrompt->end());
  std::vector<l1t::SAMuon> finalPrompt = prodMuons(muons);

  // Displace muons
  muons.clear();
  muons = *barrelDisp.product();
  muons.insert(muons.end(), forwardDisp->begin(), forwardDisp->end());
  std::vector<l1t::SAMuon> finalDisp = prodMuons(muons);

  std::unique_ptr<std::vector<l1t::SAMuon> > prompt_ptr = std::make_unique<std::vector<l1t::SAMuon> >(finalPrompt);
  std::unique_ptr<std::vector<l1t::SAMuon> > disp_ptr = std::make_unique<std::vector<l1t::SAMuon> >(finalDisp);
  iEvent.put(std::move(prompt_ptr), "prompt");
  iEvent.put(std::move(disp_ptr), "displaced");
}

DEFINE_FWK_MODULE(Phase2L1TGMTSAMuonGhostCleaner);

#endif

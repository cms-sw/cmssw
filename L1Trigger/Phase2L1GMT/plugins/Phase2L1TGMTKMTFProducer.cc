// -*- C++ -*-
//
// Package:    L1Trigger/Phase2L1GMT
// Class:      Phase2L1TGMTSAMuonProducer
// Original Author:  Michalis Bachtis

#ifndef PHASE2GMT_KMTFPRODUCER
#define PHASE2GMT_KMTFPRODUCER

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
#include "L1Trigger/Phase2L1GMT/interface/KMTF.h"
#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"

//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTKMTFProducer : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTKMTFProducer(const edm::ParameterSet&);
  ~Phase2L1TGMTKMTFProducer() override = default;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<l1t::MuonStubCollection> stubToken_;
  std::unique_ptr<KMTF> kmtf_;
  unsigned int Nprompt;
  unsigned int Ndisplaced;
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
Phase2L1TGMTKMTFProducer::Phase2L1TGMTKMTFProducer(const edm::ParameterSet& iConfig)
    : stubToken_(consumes<l1t::MuonStubCollection>(iConfig.getParameter<edm::InputTag>("stubs"))),
      kmtf_(new KMTF(iConfig.getParameter<int>("verbose"), iConfig.getParameter<edm::ParameterSet>("algo"))),
      Nprompt(iConfig.getParameter<uint>("Nprompt")),
      Ndisplaced(iConfig.getParameter<uint>("Ndisplaced")) {
  produces<std::vector<l1t::SAMuon> >("prompt").setBranchAlias("prompt");
  produces<std::vector<l1t::SAMuon> >("displaced").setBranchAlias("displaced");
  produces<std::vector<l1t::KMTFTrack> >("kmtfTracks");
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void Phase2L1TGMTKMTFProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  edm::Handle<l1t::MuonStubCollection> stubHandle;
  iEvent.getByToken(stubToken_, stubHandle);

  l1t::MuonStubRefVector stubs;
  for (uint i = 0; i < stubHandle->size(); ++i) {
    l1t::MuonStubRef stub(stubHandle, i);
    if (stub->bxNum() == 0)
      stubs.push_back(stub);
  }

  // KMTF
  std::vector<SAMuon> prompt;
  std::vector<SAMuon> displaced;
  std::pair<std::vector<l1t::KMTFTrack>, std::vector<l1t::KMTFTrack> > kmtfOutput = kmtf_->process(stubs, 0, 32);
  std::vector<l1t::KMTFTrack> kmtfTracks;
  for (const auto& track : kmtfOutput.first) {
    kmtfTracks.push_back(track);
	//convert kmtf track.zPosition (LSB_kmtf=1500./2<<16) into z0 for SAMuon. 
    //LSBSAz0 = 1.6cm is too small given only 5 bits. the range fro z0 is +- 25.6cm from DataFormats/L1TMuonPhase2/interface/Constants.h
    //eff. LSB = 1.6cm * 16 = 25.6 cm to widen the z0 range that can covered given only 5 bits.  
    //full conversion is z0_SA=z0_kmtf*(LSB_kmtf)*(1/LSB_SA)*(1/16) to convert between kmtf->SAMuon digitizations
    ap_fixed<BITSSAZ0, BITSSAZ0, AP_TRN, AP_SAT> z0 = track.zPosition() * ap_ufixed<18, 0>((1500.0 / (1 << 16)) / (LSBSAz0 * 16));
    l1t::SAMuon p(track.p4(),
                  (track.curvatureAtVertex() < 0),
                  track.ptPrompt(),
                  round(track.eta() / (M_PI / (1 << 12))),
                  track.phiAtMuon() / (1 << 5),
                  z0,
                  0,
                  track.stubs().size() - 1);
    p.setTF(l1t::tftype::bmtf);
    int bstart = 0;
    wordtype word(0);
    bstart = wordconcat<wordtype>(word, bstart, 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, p.hwCharge(), 1);
    bstart = wordconcat<wordtype>(word, bstart, p.hwPt(), BITSPT);
    bstart = wordconcat<wordtype>(word, bstart, p.hwPhi(), BITSPHI);
    bstart = wordconcat<wordtype>(word, bstart, p.hwEta(), BITSETA);
    bstart = wordconcat<wordtype>(word, bstart, p.hwD0(), BITSSAD0);
    wordconcat<wordtype>(word, bstart, track.rankPrompt(), 8);

    for (const auto& stub : track.stubs())
      p.addStub(stub);
    p.setWord(word);
    prompt.push_back(p);
  }

  for (const auto& track : kmtfOutput.second) {
    kmtfTracks.push_back(track);
    ap_int<7> dxy = track.dxy() * ap_ufixed<8, 1>(1.606);
    ap_fixed<BITSSAZ0, BITSSAZ0, AP_TRN, AP_SAT> z0 = track.zPosition() * ap_ufixed<18, 0>((1500.0 / (1 << 16)) / (LSBSAz0 * 16));
    l1t::SAMuon p(track.displacedP4(),
                  (track.curvatureAtMuon() < 0),
                  track.ptDisplaced(),
                  round(track.eta() / (M_PI / (1 << 12))),
                  track.phiAtMuon() / (1 << 5),
                  z0,
                  dxy,
                  track.approxDispChi2());
    p.setTF(l1t::tftype::bmtf);
    int bstart = 0;
    wordtype word(0);
    bstart = wordconcat<wordtype>(word, bstart, 1, 1);
    bstart = wordconcat<wordtype>(word, bstart, p.hwCharge(), 1);
    bstart = wordconcat<wordtype>(word, bstart, p.hwPt(), BITSPT);
    bstart = wordconcat<wordtype>(word, bstart, p.hwPhi(), BITSPHI);
    bstart = wordconcat<wordtype>(word, bstart, p.hwEta(), BITSETA);
    bstart = wordconcat<wordtype>(word, bstart, p.hwD0(), BITSSAD0);
    wordconcat<wordtype>(word, bstart, track.rankDisp(), 8);

    for (const auto& stub : track.stubs()) {
      p.addStub(stub);
    }
    p.setWord(word);
    displaced.push_back(p);
  }
  std::unique_ptr<std::vector<l1t::SAMuon> > prompt_ptr = std::make_unique<std::vector<l1t::SAMuon> >(prompt);
  std::unique_ptr<std::vector<l1t::SAMuon> > displaced_ptr = std::make_unique<std::vector<l1t::SAMuon> >(displaced);
  std::unique_ptr<std::vector<l1t::KMTFTrack> > kmtf_ptr = std::make_unique<std::vector<l1t::KMTFTrack> >(kmtfTracks);
  iEvent.put(std::move(prompt_ptr), "prompt");
  iEvent.put(std::move(displaced_ptr), "displaced");
  iEvent.put(std::move(kmtf_ptr), "kmtfTracks");
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTKMTFProducer);
#endif

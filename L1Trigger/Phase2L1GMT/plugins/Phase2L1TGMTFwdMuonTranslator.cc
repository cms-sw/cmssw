// -*- C++ -*-

#ifndef PHASE2GMT_SAFWDMUONTRANSLATOR
#define PHASE2GMT_SAFWDMUONTRANSLATOR

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

//
// class declaration
//
using namespace Phase2L1GMT;
using namespace l1t;

class Phase2L1TGMTFwdMuonTranslator : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1TGMTFwdMuonTranslator(const edm::ParameterSet&);
  ~Phase2L1TGMTFwdMuonTranslator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  l1t::SAMuon Convertl1tMuon(const l1t::RegionalMuonCand& mu, const int bx_, bool isDisplaced = false);
  l1t::MuonStubRefVector selectLayerBX(const l1t::MuonStubRefVector& all, int bx, uint layer);
  void associateStubs(l1t::SAMuon&, const l1t::MuonStubRefVector&);

  // ----------member data ---------------------------
  edm::EDGetTokenT<l1t::MuonStubCollection> stubToken_;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> omtfTrackToken_;
  edm::EDGetTokenT<l1t::phase2::EMTFTrackCollection> emtfTrackToken_;
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
Phase2L1TGMTFwdMuonTranslator::Phase2L1TGMTFwdMuonTranslator(const edm::ParameterSet& iConfig)
<<<<<<< HEAD
    : muonToken_(consumes<l1t::MuonBxCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      stubToken_(consumes<l1t::MuonStubCollection>(iConfig.getParameter<edm::InputTag>("stubs"))) {
=======
    : stubToken_(consumes<l1t::MuonStubCollection>(iConfig.getParameter<edm::InputTag>("stubs"))),
      omtfTrackToken_(consumes<l1t::RegionalMuonCandBxCollection>(iConfig.getParameter<edm::InputTag>("omtfTracks"))),
      emtfTrackToken_(consumes<l1t::phase2::EMTFTrackCollection>(iConfig.getParameter<edm::InputTag>("emtfTracks"))) {
>>>>>>> d34010dded1 (Add Phase-2 OMTF and EMTF to GMT SA)
  produces<std::vector<l1t::SAMuon> >("prompt").setBranchAlias("prompt");
}

// ------------ method called to produce the data  ------------
void Phase2L1TGMTFwdMuonTranslator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
<<<<<<< HEAD
  edm::Handle<l1t::MuonBxCollection> muon;
  iEvent.getByToken(muonToken_, muon);
  edm::Handle<l1t::MuonStubCollection> stubHandle;
  iEvent.getByToken(stubToken_, stubHandle);
=======

  edm::Handle<l1t::MuonStubCollection> stubHandle;
  iEvent.getByToken(stubToken_, stubHandle);

  edm::Handle<l1t::phase2::EMTFTrackCollection> emtf_tracks;
  iEvent.getByToken(emtfTrackToken_, emtf_tracks);

  edm::Handle<l1t::RegionalMuonCandBxCollection> omtf_tracks;
  iEvent.getByToken(omtfTrackToken_, omtf_tracks);

  // Process Stubs
>>>>>>> d34010dded1 (Add Phase-2 OMTF and EMTF to GMT SA)
  l1t::MuonStubRefVector stubs;
  for (uint i = 0; i < stubHandle->size(); ++i) {
    l1t::MuonStubRef stub(stubHandle, i);
    if (stub->bxNum() == 0)
      stubs.push_back(stub);
  }

  std::vector<SAMuon> prompt;

  //  TODO: Will receive hybrid stubs from OMTF/EMTF
  //  std::vector<l1t::MuonStub> hybridStubs;
  //  edm::RefProd stubRefProd = iEvent.getRefBeforePut<std::vector<l1t::MuonStub> >("hybridStubs");
  //  l1t::MuonStubRef::key_type idxStub =0;

<<<<<<< HEAD
  for (unsigned int i = 0; i < muon->size(0); ++i) {
    const l1t::Muon& mu = muon->at(0, i);
    l1t::SAMuon samuon = Convertl1tMuon(mu, 0);
    if (samuon.tfType() == l1t::tftype::bmtf)
      continue;
=======
  // Convert OMTF Muons to SAMuons
  for (unsigned int i = 0; i < omtf_tracks->size(0); ++i) {
    const l1t::RegionalMuonCand& mu = omtf_tracks->at(0, i);
    // Since OMTF is using Phase-1 LSB, will convert to SAMuon locally
    // We should move to passing words in future
    l1t::SAMuon samuon;
    if (mu.hwPt() > 0)
      samuon = Convertl1tMuon(mu, 0);
    else if (mu.hwPtUnconstrained() > 0)  //Assume exculsive, need double check
      samuon = Convertl1tMuon(mu, 0, true);

>>>>>>> d34010dded1 (Add Phase-2 OMTF and EMTF to GMT SA)
    //now associate the stubs
    associateStubs(samuon, stubs);

    // Add To Collections
    if (mu.hwPt() > 0)
      prompt.push_back(samuon);
    else if (mu.hwPtUnconstrained() > 0)
      displaced.push_back(samuon);
  }
  std::unique_ptr<std::vector<l1t::SAMuon> > prompt_ptr = std::make_unique<std::vector<l1t::SAMuon> >(prompt);
  iEvent.put(std::move(prompt_ptr), "prompt");
}

// ===  FUNCTION  ============================================================
//         Name:  Phase2L1TGMTFwdMuonTranslator::Convertl1tMuon
//  Description:
// ===========================================================================
SAMuon Phase2L1TGMTFwdMuonTranslator::Convertl1tMuon(const l1t::RegionalMuonCand& mu, const int bx_, bool isDisplaced) {
  ap_uint<BITSSAQUAL> qual = mu.hwQual();
  int charge = mu.hwSign();

  ap_uint<BITSGTPT> pt = 0;
  if (!isDisplaced && mu.hwPt() > 0)
    pt = round(mu.hwPt() * 0.5 / LSBpt);  // Phase-1 LSB 0.5GeV
  if (isDisplaced && mu.hwPtUnconstrained() > 0)
    pt = round(mu.hwPtUnconstrained() * 0.5 / LSBpt);  // Phase-1 LSB 0.5GeV

  constexpr double p1phiLSB = 2 * M_PI / 576;
  ap_int<BITSGTPHI> phi = round(mu.hwPhi() * p1phiLSB / LSBphi);  // Phase-1 LSB (2*pi/576)
  ap_int<BITSGTETA> eta = round(mu.hwEta() * 0.010875 / LSBeta);  // Phase-1 LSB 0.010875

  // FIXME: Below are not well defined in phase1 GMT
  // Using the version from Correlator for now
  ap_int<BITSSAZ0> z0 = 0;  // No tracks info in Phase 1
  // Use 2 bits with LSB = 30cm for BMTF and 25cm for EMTF currently, but subjet to change
  ap_int<BITSSAD0> d0 = mu.hwDXY();

  //Here do not use the word format to GT but use the word format expected by GMT
  int bstart = 0;
  wordtype word(0);
  bstart = wordconcat<wordtype>(word, bstart, 1, 1);
  bstart = wordconcat<wordtype>(word, bstart, pt, BITSGTPT);
  bstart = wordconcat<wordtype>(word, bstart, phi, BITSGTPHI);
  bstart = wordconcat<wordtype>(word, bstart, eta, BITSGTETA);
  bstart = wordconcat<wordtype>(word, bstart, z0, BITSSAZ0);
  bstart = wordconcat<wordtype>(word, bstart, d0, BITSSAD0);
  bstart = wordconcat<wordtype>(word, bstart, charge, 1);
  bstart = wordconcat<wordtype>(word, bstart, qual, BITSSAQUAL);

  // Calculate Lorentz Vector
  math::PtEtaPhiMLorentzVector p4(pt * LSBpt, eta * LSBeta, phi * LSBphi, 0.0);
  SAMuon samuon(p4, charge, pt.to_uint(), eta.to_int(), phi.to_int(), z0.to_int(), d0.to_int(), qual.to_uint());
  samuon.setTF(mu.trackFinderType());
  samuon.setWord(word);

  return samuon;
}  // -----  end of function Phase2L1TGMTFwdMuonTranslator::Convertl1tMuon  -----

l1t::MuonStubRefVector Phase2L1TGMTFwdMuonTranslator::selectLayerBX(const l1t::MuonStubRefVector& all,
                                                                    int bx,
                                                                    uint layer) {
  l1t::MuonStubRefVector out;
  for (const auto& stub : all) {
    if (stub->bxNum() == bx && stub->tfLayer() == layer)
      out.push_back(stub);
  }
  return out;
}

void Phase2L1TGMTFwdMuonTranslator::associateStubs(l1t::SAMuon& mu, const l1t::MuonStubRefVector& stubs) {
  for (unsigned int layer = 0; layer <= 4; ++layer) {
    l1t::MuonStubRefVector selectedStubs = selectLayerBX(stubs, 0, layer);
    int bestStubINT = -1;
    float dPhi = 1000.0;
    for (uint i = 0; i < selectedStubs.size(); ++i) {
      const l1t::MuonStubRef& stub = selectedStubs[i];
      float deltaPhi =
          (stub->quality() & 0x1) ? stub->offline_coord1() - mu.p4().phi() : stub->offline_coord2() - mu.p4().phi();
      if (deltaPhi > M_PI)
        deltaPhi = deltaPhi - 2 * M_PI;
      if (deltaPhi < -M_PI)
        deltaPhi = deltaPhi + 2 * M_PI;
      deltaPhi = fabs(deltaPhi);
      float deltaEta = (stub->etaQuality() == 0 || (stub->etaQuality() & 0x1))
                           ? fabs(stub->offline_eta1() - mu.p4().eta())
                           : fabs(stub->offline_eta2() - mu.p4().eta());
      if (deltaPhi < 0.3 && deltaEta < 0.3 && deltaPhi < dPhi) {
        dPhi = deltaPhi;
        bestStubINT = i;
      }
    }
    if (bestStubINT >= 0) {
      mu.addStub(selectedStubs[bestStubINT]);
    }
  }
}

SAMuon Phase2L1TGMTFwdMuonTranslator::ConvertEMTFTrack(const l1t::phase2::EMTFTrack& track, const int bx_) {
  // Convert EMTF Phi and Theta to Global Phi and Eta
  float track_phi = emtf::phase2::tp::calc_phi_glob_rad_from_loc(
      track.sector(), emtf::phase2::tp::calc_phi_loc_deg_from_int(track.modelPhi()));
  float track_theta = emtf::phase2::tp::calc_theta_rad_from_int(track.modelEta());
  float track_eta = -1 * std::log(std::tan(track_theta / 2));

  track_theta *= track.endcap();
  track_eta *= track.endcap();

  // Calculate Lorentz Vector
  // Muon mass taken from L1Trigger/L1TMuon/plugins/L1TMuonProducer.cc
  math::PtEtaPhiMLorentzVector p4(track.emtfPt() * LSBpt, track_eta, track_phi, 0.0);

  // Quantize Values
  ap_uint<BITSSAQUAL> qual = track.emtfModeV2();  // Not sure how this should be handled; using mode for now
  int charge = track.emtfQ();                     // EMTF uses the same convention
  ap_uint<BITSGTPT> pt = track.emtfPt();          // Quantized by EMTF in the same units
  ap_int<BITSGTPHI> phi = round(track_phi / LSBphi);
  ap_int<BITSGTETA> eta = round(track_eta / LSBeta);
  ap_int<BITSSAZ0> z0 = track.emtfZ0();  // Quantized by EMTF in the same units
  ap_int<BITSSAD0> d0 = track.emtfD0();  // Quantized by EMTF in the same units

  //Here do not use the word format to GT but use the word format expected by GMT
  int bstart = 0;
  wordtype word(0);
  bstart = wordconcat<wordtype>(word, bstart, 1, 1);
  bstart = wordconcat<wordtype>(word, bstart, pt, BITSPT);
  bstart = wordconcat<wordtype>(word, bstart, phi, BITSPHI);
  bstart = wordconcat<wordtype>(word, bstart, eta, BITSETA);
  bstart = wordconcat<wordtype>(word, bstart, z0, BITSSAZ0);
  bstart = wordconcat<wordtype>(word, bstart, d0, BITSSAD0);
  bstart = wordconcat<wordtype>(word, bstart, charge, 1);
  bstart = wordconcat<wordtype>(word, bstart, qual, BITSSAQUAL);

  SAMuon samuon(p4, charge, pt.to_uint(), eta.to_int(), phi.to_int(), z0.to_int(), d0.to_int(), qual.to_uint());

  // +1=Positive Endcap and -1=Negative Endcap
  if (track.endcap() == 1)
    samuon.setTF(tftype::emtf_pos);
  else
    samuon.setTF(tftype::emtf_neg);

  samuon.setWord(word);

  return samuon;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void Phase2L1TGMTFwdMuonTranslator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;

  // Input Collections
  desc.add<edm::InputTag>("stubs", edm::InputTag("gmtStubs"));
  desc.add<edm::InputTag>("emtfTracks", edm::InputTag("simEmtfDigisPhase2"));
  desc.add<edm::InputTag>("omtfTracks", edm::InputTag("simOmtfPhase2Digis"));

  // Register
  descriptions.add("Phase2L1TGMTFwdMuonTranslator", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(Phase2L1TGMTFwdMuonTranslator);
#endif

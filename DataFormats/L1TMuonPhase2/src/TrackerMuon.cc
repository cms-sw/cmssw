
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

using namespace l1t;

TrackerMuon::TrackerMuon() : hwZ0_(0), hwD0_(0) {}

TrackerMuon::TrackerMuon(
    const edm::Ptr<L1TTTrackType>& trk, bool charge, uint pt, int eta, int phi, int z0, int d0, uint quality)
    : L1Candidate(LorentzVector(trk->momentum().x(), trk->momentum().y(), trk->momentum().z(), trk->momentum().mag()),
                  pt,
                  eta,
                  phi,
                  quality),
      trkPtr_(trk),
      hwCharge_(charge),
      hwZ0_(z0),
      hwD0_(d0),
      hwBeta_(15) {}

TrackerMuon::~TrackerMuon() {}

void TrackerMuon::print() const {
  LogDebug("TrackerMuon") << "Tracker Muon : charge=" << hwCharge_ << " pt=" << hwPt() << "," << p4().pt()
                          << " eta=" << hwEta() << "," << p4().eta() << " phi=" << hwPhi() << "," << p4().phi()
                          << " z0=" << hwZ0_ << " d0=" << hwD0_ << " isolation=" << hwIso() << " beta=" << hwBeta_
                          << " quality=" << hwQual();
}

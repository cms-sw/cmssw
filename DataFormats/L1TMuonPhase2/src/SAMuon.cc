#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace l1t;

SAMuon::SAMuon() : hwZ0_(0), hwD0_(0), word_(0) {}

SAMuon::SAMuon(const l1t::Muon& mu, bool charge, uint pt, int eta, int phi, int z0, int d0, uint quality)
    : L1Candidate(mu.p4(), pt, eta, phi, quality), hwCharge_(charge), hwZ0_(z0), hwD0_(d0), word_(0) {}

SAMuon::~SAMuon() {}

void SAMuon::print() const {
  LogDebug("SAMuon") << "Standalone Muon: charge=" << hwCharge_ << " pt=" << hwPt() << "," << p4().pt()
                     << " eta=" << hwEta() << "," << p4().eta() << " phi=" << hwPhi() << "," << p4().phi()
                     << " z0=" << hwZ0_ << " d0=" << hwD0_ << " isolation=" << hwIso() << " beta=" << hwBeta_
                     << " quality=" << hwQual();
}

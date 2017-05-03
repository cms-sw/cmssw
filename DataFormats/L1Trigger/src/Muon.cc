#include "DataFormats/L1Trigger/interface/Muon.h"

l1t::Muon::Muon()
  : L1Candidate(math::PtEtaPhiMLorentzVector{0., 0., 0., 0.}, 0., 0., 0., 0, 0),
    hwCharge_(0),
    hwChargeValid_(0),
    tfMuonIndex_(-1),
    hwTag_(0),
    debug_(false),
    hwIsoSum_(0),
    hwDPhiExtra_(0),
    hwDEtaExtra_(0),
    hwRank_(0),
    hwEtaAtVtx_(0),
    hwPhiAtVtx_(0),
    etaAtVtx_(0.),
    phiAtVtx_(0.)
{

}

l1t::Muon::Muon( const LorentzVector& p4,
    int pt,
    int eta,
    int phi,
    int qual,
    int charge,
    int chargeValid,
    int iso,
    int tfMuonIndex,
    int tag,
    bool debug,
    int isoSum,
    int dPhi,
    int dEta,
    int rank,
    int hwEtaAtVtx,
    int hwPhiAtVtx,
    double etaAtVtx,
    double phiAtVtx)
  : L1Candidate(p4, pt, eta, phi, qual, iso),
    hwCharge_(charge),
    hwChargeValid_(chargeValid),
    tfMuonIndex_(tfMuonIndex),
    hwTag_(tag),
    debug_(debug),
    hwIsoSum_(isoSum),
    hwDPhiExtra_(dPhi),
    hwDEtaExtra_(dEta),
    hwRank_(rank),
    hwEtaAtVtx_(hwEtaAtVtx),
    hwPhiAtVtx_(hwPhiAtVtx),
    etaAtVtx_(etaAtVtx),
    phiAtVtx_(phiAtVtx)
{
  
}

l1t::Muon::Muon( const PolarLorentzVector& p4,
    int pt,
    int eta,
    int phi,
    int qual,
    int charge,
    int chargeValid,
    int iso,
    int tfMuonIndex,
    int tag,
    bool debug,
    int isoSum,
    int dPhi,
    int dEta,
    int rank,
    int hwEtaAtVtx,
    int hwPhiAtVtx,
    double etaAtVtx,
    double phiAtVtx)
  : L1Candidate(p4, pt, eta, phi, qual, iso),
    hwCharge_(charge),
    hwChargeValid_(chargeValid),
    tfMuonIndex_(tfMuonIndex),
    hwTag_(tag),
    debug_(debug),
    hwIsoSum_(isoSum),
    hwDPhiExtra_(dPhi),
    hwDEtaExtra_(dEta),
    hwRank_(rank),
    hwEtaAtVtx_(hwEtaAtVtx),
    hwPhiAtVtx_(hwPhiAtVtx),
    etaAtVtx_(etaAtVtx),
    phiAtVtx_(phiAtVtx)
{
  
}

l1t::Muon::~Muon() 
{

}


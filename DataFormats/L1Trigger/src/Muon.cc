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

void 
l1t::Muon::setHwCharge(int charge)
{
  hwCharge_ = charge;
}

void 
l1t::Muon::setHwChargeValid(int valid)
{
  hwChargeValid_ = valid;
}

void 
l1t::Muon::setHwTag(int tag)
{
  hwTag_ = tag;
}

void 
l1t::Muon::setTfMuonIndex(int index)
{
  tfMuonIndex_ = index;
}

void
l1t::Muon::setHwEtaAtVtx(int hwEtaAtVtx)
{
  hwEtaAtVtx_ = hwEtaAtVtx;
}

void
l1t::Muon::setHwPhiAtVtx(int hwPhiAtVtx)
{
  hwPhiAtVtx_ = hwPhiAtVtx;
}

void
l1t::Muon::setEtaAtVtx(double etaAtVtx)
{
  etaAtVtx_ = etaAtVtx;
}

void
l1t::Muon::setPhiAtVtx(double phiAtVtx)
{
  phiAtVtx_ = phiAtVtx;
}

void
l1t::Muon::setHwIsoSum(int isoSum) 
{
  hwIsoSum_ = isoSum;
}

void
l1t::Muon::setHwDPhiExtra(int dPhi)
{
  hwDPhiExtra_ = dPhi;
}

void
l1t::Muon::setHwDEtaExtra(int dEta) 
{
  hwDEtaExtra_ = dEta;
}

void
l1t::Muon::setHwRank(int rank) 
{
  hwRank_ = rank;
}

void
l1t::Muon::setDebug(bool debug)
{
  debug_ = debug;
}

int 
l1t::Muon::hwCharge() const
{
  return hwCharge_;
}

int 
l1t::Muon::hwChargeValid() const
{
  return hwChargeValid_;
}

int 
l1t::Muon::hwTag() const
{
  return hwTag_;
}

int 
l1t::Muon::tfMuonIndex() const
{
  return tfMuonIndex_;
}

int
l1t::Muon::hwEtaAtVtx() const
{
  return hwEtaAtVtx_;
}

int
l1t::Muon::hwPhiAtVtx() const
{
  return hwPhiAtVtx_;
}

double
l1t::Muon::etaAtVtx() const
{
  return etaAtVtx_;
}

double
l1t::Muon::phiAtVtx() const
{
  return phiAtVtx_;
}

int
l1t::Muon::hwIsoSum() const 
{
  return hwIsoSum_;
}

int
l1t::Muon::hwDPhiExtra() const
{
  return hwDPhiExtra_;
}

int
l1t::Muon::hwDEtaExtra() const
{
  return hwDEtaExtra_;
}

int
l1t::Muon::hwRank() const
{
  return hwRank_;
}

bool
l1t::Muon::debug() const
{
  return debug_;
}

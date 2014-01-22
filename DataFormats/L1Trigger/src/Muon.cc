
#include "DataFormats/L1Trigger/interface/Muon.h"

l1t::Muon::Muon( const LorentzVector& p4,
		 int pt,
		 int eta,
		 int phi,
		 int qual,
		 int charge,
		 int chargeValid,
		 int iso,
		 int mip,
		 int tag )
  : L1Candidate(p4, pt, eta, phi, qual),
    hwCharge_(charge),
    hwChargeValid_(chargeValid),
    hwIso_(iso),
    hwMip_(mip),
    hwTag_(tag)
{
  
}

l1t::Muon::~Muon() 
{

}

void l1t::Muon::setHwCharge(int charge)
{
  hwCharge_ = charge;
}

void l1t::Muon::setHwChargeValid(int valid)
{
  hwChargeValid_ = valid;
}

void l1t::Muon::setHwIso(int iso)
{
  hwIso_ = iso;
}

void l1t::Muon::setHwMip(int mip)
{
  hwMip_ = mip;
}

void l1t::Muon::setHwTag(int tag)
{
  hwTag_ = tag;
}


int l1t::Muon::hwCharge() const
{
  return hwCharge_;
}

int l1t::Muon::hwChargeValid() const
{
  return hwChargeValid_;
}

int l1t::Muon::hwIso() const
{
  return hwIso_;
}

int l1t::Muon::hwMip() const
{
  return hwMip_;
}

int l1t::Muon::hwTag() const
{
  return hwTag_;
}

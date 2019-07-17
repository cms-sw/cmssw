#include "DataFormats/L1Trigger/interface/MuonPhase2.h"

l1t::MuonPhase2::MuonPhase2()
  : Muon()
  , ptNoVtx_(-1)
{

}

l1t::MuonPhase2::MuonPhase2( const l1t::MuonPhase2& muon)
  : l1t::Muon( muon )
{
  ptNoVtx_ = muon.getPtNoVtx();
}

l1t::MuonPhase2::MuonPhase2( const l1t::Muon& muon)
  : l1t::Muon( muon )
{
  ptNoVtx_ = -1;
}

l1t::MuonPhase2::~MuonPhase2()
{

}


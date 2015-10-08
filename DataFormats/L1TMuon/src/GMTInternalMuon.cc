#include "DataFormats/L1TMuon/interface/GMTInternalMuon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
//#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"


namespace l1t {

GMTInternalMuon::GMTInternalMuon(const GMTInternalMuon& other) :
  m_regional(other.m_regional), m_hwRank(other.m_hwRank), m_hwCancelBit(other.m_hwCancelBit), m_hwIsoSum(other.m_hwIsoSum),
  m_hwDeltaEta(other.m_hwDeltaEta), m_hwDeltaPhi(other.m_hwDeltaPhi), m_hwAbsIso(other.m_hwAbsIso), m_hwRelIso(other.m_hwRelIso),
  m_hwGlobalPhi(other.m_hwGlobalPhi), m_hwCaloIndex(other.m_hwCaloIndex) {

}

GMTInternalMuon::GMTInternalMuon(const RegionalMuonCand& regional, int globalPhi) :
  m_regional(regional), m_hwRank(0), m_hwCancelBit(0), m_hwIsoSum(0), m_hwDeltaEta(0), m_hwDeltaPhi(0), m_hwAbsIso(-1), m_hwRelIso(-1), m_hwGlobalPhi(globalPhi)
{

}

void
GMTInternalMuon::setExtrapolation(int deta, int dphi)
{
  m_hwDeltaEta = deta;
  m_hwDeltaPhi = dphi;
}


} // namespace l1t

#include "../interface/MicroGMTIsolationUnit.h"

#include "DataFormats/L1TMuon/interface/GMTInputCaloSum.h"
#include "DataFormats/L1TMuon/interface/GMTInternalMuon.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


l1t::MicroGMTIsolationUnit::MicroGMTIsolationUnit (const edm::ParameterSet& iConfig) :
  m_BEtaExtrapolation(iConfig, "BEtaExtrapolationLUTSettings", 0), m_BPhiExtrapolation(iConfig, "BPhiExtrapolationLUTSettings", 1), m_OEtaExtrapolation(iConfig, "OEtaExtrapolationLUTSettings", 0),
  m_OPhiExtrapolation(iConfig, "OPhiExtrapolationLUTSettings", 1), m_FEtaExtrapolation(iConfig, "FEtaExtrapolationLUTSettings", 0), m_FPhiExtrapolation(iConfig, "FPhiExtrapolationLUTSettings", 1),
  m_IdxSelMemEta(iConfig, "IdxSelMemEtaLUTSettings", 0), m_IdxSelMemPhi(iConfig, "IdxSelMemPhiLUTSettings", 1), m_RelIsoCheckMem(iConfig, "RelIsoCheckMemLUTSettings"),
  m_AbsIsoCheckMem(iConfig, "AbsIsoCheckMemLUTSettings"), m_initialSums(false)
{
  m_etaExtrapolationLUTs[tftype::bmtf] = &m_BEtaExtrapolation;
  m_phiExtrapolationLUTs[tftype::bmtf] = &m_BPhiExtrapolation;
  m_etaExtrapolationLUTs[tftype::omtf_pos] = &m_OEtaExtrapolation;
  m_etaExtrapolationLUTs[tftype::omtf_neg] = &m_OEtaExtrapolation;
  m_phiExtrapolationLUTs[tftype::omtf_pos] = &m_OPhiExtrapolation;
  m_phiExtrapolationLUTs[tftype::omtf_neg] = &m_OPhiExtrapolation;
  m_etaExtrapolationLUTs[tftype::emtf_pos] = &m_FEtaExtrapolation;
  m_etaExtrapolationLUTs[tftype::emtf_neg] = &m_FEtaExtrapolation;
  m_phiExtrapolationLUTs[tftype::emtf_pos] = &m_FPhiExtrapolation;
  m_phiExtrapolationLUTs[tftype::emtf_neg] = &m_FPhiExtrapolation;
}

l1t::MicroGMTIsolationUnit::~MicroGMTIsolationUnit ()
{
}

int
l1t::MicroGMTIsolationUnit::getCaloIndex(MicroGMTConfiguration::InterMuon& mu) const
{
  // handle the wrap-around of phi:
  int phi = (mu.hwGlobalPhi() + mu.hwDPhi())%576;
  if (phi < 0) {
    phi = 576+phi;
  }

  int phiIndex = m_IdxSelMemPhi.lookup(phi);
  int eta = mu.hwEta()+mu.hwDEta();
  eta = MicroGMTConfiguration::getTwosComp(eta, 9);
  int etaIndex = m_IdxSelMemEta.lookup(eta);
  mu.setHwCaloEta(etaIndex);
  mu.setHwCaloPhi(phiIndex);

  return phiIndex + etaIndex*36;
}

void
l1t::MicroGMTIsolationUnit::extrapolateMuons(MicroGMTConfiguration::InterMuonList& inputmuons) const {
  for (auto &mu : inputmuons) {
    // only use 6 LSBs of pt:
    int ptRed = mu->hwPt() & 0b111111;
    // here we drop the two LSBs and masking the MSB
    int etaAbsRed = (std::abs(mu->hwEta()) >> 2) & ((1 << 6) - 1);

    int deltaPhi = 0;
    int deltaEta = 0;

    if (mu->hwPt() < 64) { // extrapolation only for "low" pT muons
      int sign = 1;
      if (mu->hwSign() == 0) {
        sign = -1;
      }
      deltaPhi = (m_phiExtrapolationLUTs.at(mu->trackFinderType())->lookup(etaAbsRed, ptRed) << 3) * sign;
      deltaEta = (m_etaExtrapolationLUTs.at(mu->trackFinderType())->lookup(etaAbsRed, ptRed) << 3);
    }

    mu->setExtrapolation(deltaEta, deltaPhi);
  }
}

void
l1t::MicroGMTIsolationUnit::calculate5by1Sums(const MicroGMTConfiguration::CaloInputCollection& inputs, int bx)
{
  m_5by1TowerSums.clear();
  if (inputs.size(bx) == 0) return;

  for (int iphi = 0; iphi < 36; ++iphi) {
    int iphiIndexOffset = iphi*28;
    m_5by1TowerSums.push_back(inputs.at(bx, iphiIndexOffset).etBits()+inputs.at(bx, iphiIndexOffset+1).etBits()+inputs.at(bx, iphiIndexOffset+2).etBits());//ieta = 0 (tower -28)
    m_5by1TowerSums.push_back(inputs.at(bx, iphiIndexOffset-1).etBits()+inputs.at(bx, iphiIndexOffset).etBits()+inputs.at(bx, iphiIndexOffset+1).etBits()+inputs.at(bx, iphiIndexOffset+2).etBits()); //
    for (int ieta = 2; ieta < 26; ++ieta) {
      int sum = 0;
      for (int dIEta = -2; dIEta <= 2; ++dIEta) {
        sum += inputs.at(bx, iphiIndexOffset+dIEta).etBits();
      }
      m_5by1TowerSums.push_back(sum);
    }
    m_5by1TowerSums.push_back(inputs.at(bx, iphiIndexOffset+1).etBits()+inputs.at(bx, iphiIndexOffset).etBits()+inputs.at(bx, iphiIndexOffset-1).etBits()+inputs.at(bx, iphiIndexOffset-2).etBits());
    m_5by1TowerSums.push_back(inputs.at(bx, iphiIndexOffset).etBits()+inputs.at(bx, iphiIndexOffset-1).etBits()+inputs.at(bx, iphiIndexOffset-2).etBits());//ieta = 0 (tower 28)
  }

  m_initialSums = true;
}


int
l1t::MicroGMTIsolationUnit::calculate5by5Sum(unsigned index) const
{
  if (index > m_5by1TowerSums.size()) {
    edm::LogWarning("energysum out of bounds!");
    return 0;
  }
  // phi wrap around:
  int returnSum = 0;
  for (int dIPhi = -2; dIPhi <= 2; ++dIPhi) {
    int currIndex = (index + dIPhi*28)%1008; // wrap-around at top
    if (currIndex < 0) currIndex = 1008+currIndex;
    if ((unsigned)currIndex < m_5by1TowerSums.size()) {
      returnSum += m_5by1TowerSums[currIndex];
    } else {
      edm::LogWarning("energysum out of bounds!");
    }
  }
  return std::min(31, returnSum);
}

void
l1t::MicroGMTIsolationUnit::isolate(MicroGMTConfiguration::InterMuonList& muons) const
{
  for (auto& mu : muons) {
    int caloIndex = getCaloIndex(*mu);
    int energySum = calculate5by5Sum(caloIndex);
    mu->setHwIsoSum(energySum);

    int absIso = m_AbsIsoCheckMem.lookup(energySum);
    int relIso = m_RelIsoCheckMem.lookup(energySum, mu->hwPt());

    mu->setHwRelIso(relIso);
    mu->setHwAbsIso(absIso);
  }
}

void l1t::MicroGMTIsolationUnit::setTowerSums(const MicroGMTConfiguration::CaloInputCollection& inputs, int bx) {
  m_towerEnergies.clear();
  if (inputs.size(bx) == 0) return;
  for (auto input = inputs.begin(bx); input != inputs.end(bx); ++input) {
    if ( input->etBits() != 0 ) {
      m_towerEnergies[input->hwEta()*36+input->hwPhi()] = input->etBits();
    }
  }

  m_initialSums = true;

}

void l1t::MicroGMTIsolationUnit::isolatePreSummed(MicroGMTConfiguration::InterMuonList& muons) const
{
  for (auto mu : muons) {
    int caloIndex = getCaloIndex(*mu);
    int energySum = 0;
    if (m_towerEnergies.count(caloIndex) == 1) {
      energySum = m_towerEnergies.at(caloIndex);
    }

    mu->setHwIsoSum(energySum);

    int absIso = m_AbsIsoCheckMem.lookup(energySum);
    int relIso = m_RelIsoCheckMem.lookup(energySum, mu->hwPt());

    mu->setHwRelIso(relIso);
    mu->setHwAbsIso(absIso);
  }

}

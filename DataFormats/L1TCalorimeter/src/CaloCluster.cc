
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

namespace l1t::io_v1 {
  CaloCluster::CaloCluster(const LorentzVector p4, int pt, int eta, int phi)
      : L1Candidate(p4, pt, eta, phi),
        m_clusterFlags(0x7FF)  // first 11 flags at 1
  {}

  CaloCluster::~CaloCluster() {}

  void CaloCluster::setClusterFlag(ClusterFlag flag, bool val) {
    if (val) {
      m_clusterFlags |= (0x1 << flag);
    } else {
      m_clusterFlags &= ~(0x1 << flag);
    }
  };

  void CaloCluster::setHwPtEm(int pt) { m_hwPtEm = pt; }

  void CaloCluster::setHwPtHad(int pt) { m_hwPtHad = pt; }

  void CaloCluster::setHwSeedPt(int pt) { m_hwSeedPt = pt; }

  void CaloCluster::setFgEta(int fgEta) { m_fgEta = fgEta; }

  void CaloCluster::setFgPhi(int fgPhi) { m_fgPhi = fgPhi; }

  void CaloCluster::setHOverE(int hOverE) { m_hOverE = hOverE; }

  void CaloCluster::setFgECAL(int fgECAL) { m_fgECAL = fgECAL; }

  bool CaloCluster::checkClusterFlag(ClusterFlag flag) const { return (m_clusterFlags & (0x1 << flag)); };

  bool CaloCluster::isValid() const { return (checkClusterFlag(INCLUDE_SEED)); }

  int CaloCluster::hwPtEm() const { return m_hwPtEm; }

  int CaloCluster::hwPtHad() const { return m_hwPtHad; }

  int CaloCluster::hwSeedPt() const { return m_hwSeedPt; }

  int CaloCluster::fgEta() const { return m_fgEta; }

  int CaloCluster::fgPhi() const { return m_fgPhi; }

  int CaloCluster::hOverE() const { return m_hOverE; }

  int CaloCluster::fgECAL() const { return m_fgECAL; }

  bool CaloCluster::operator<(const CaloCluster& cl) const {
    bool res = false;
    // Favour high pT
    if (hwPt() < cl.hwPt())
      res = true;
    else if (hwPt() == cl.hwPt()) {
      // Favour central clusters
      if (abs(hwEta()) > abs(cl.hwEta()))
        res = true;
      else if (abs(hwEta()) == abs(cl.hwEta())) {
        // Favour small phi (arbitrary)
        if (hwPhi() > cl.hwPhi())
          res = true;
      }
    }
    return res;
  }
}  // namespace l1t::io_v1

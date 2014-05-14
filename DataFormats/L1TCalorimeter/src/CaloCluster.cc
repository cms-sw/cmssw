
#include "DataFormats/L1TCalorimeter/interface/CaloCluster.h"

l1t::CaloCluster::CaloCluster( const LorentzVector p4, 
			     int pt,
			     int eta,
			     int phi)
  : L1Candidate(p4, pt, eta, phi),
    m_clusterFlags(0)
{
  
}

l1t::CaloCluster::~CaloCluster() 
{
  
}

void l1t::CaloCluster::setClusterFlag(ClusterFlag flag, bool val)
{
  if(val) 
  {
    m_clusterFlags |= (0x1<<flag);
  }
  else
  {
    m_clusterFlags &= ~(0x1<<flag);
  }
};

void l1t::CaloCluster::setHwSeedPt(int pt)
{
  m_hwSeedPt = pt;
}

void l1t::CaloCluster::setHOverE(int hOverE)
{
  m_hOverE = hOverE;
}

bool l1t::CaloCluster::checkClusterFlag(ClusterFlag flag) const 
{
  return (m_clusterFlags & (0x1<<flag));
};

bool l1t::CaloCluster::isValid() const
{
    return ( checkClusterFlag(PASS_THRES_SEED) && checkClusterFlag(PASS_FILTER_CLUSTER) );
}

int l1t::CaloCluster::hwSeedPt() const
{
  return m_hwSeedPt;
}

int l1t::CaloCluster::fgEta() const
{
  return m_fgEta;
}

int l1t::CaloCluster::fgPhi() const
{
  return m_fgPhi;
}

int l1t::CaloCluster::hOverE() const
{
  return m_hOverE;
}

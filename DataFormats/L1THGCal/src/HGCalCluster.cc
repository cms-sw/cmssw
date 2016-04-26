#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;

HGCalCluster::HGCalCluster( const LorentzVector p4, 
			     int pt,
			     int eta,
			     int phi)
  : L1Candidate(p4, pt, eta, phi),
    m_clusterFlags(0x7FF) // first 11 flags at 1
{
  
}

HGCalCluster::~HGCalCluster() 
{
  
}

void HGCalCluster::setClusterFlag(ClusterFlag flag, bool val)
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

void HGCalCluster::setHwPtEm(int pt)
{
  m_hwPtEm = pt;
}

void HGCalCluster::setHwPtHad(int pt)
{
  m_hwPtHad = pt;
}

void HGCalCluster::setHwSeedPt(int pt)
{
  m_hwSeedPt = pt;
}

void HGCalCluster::setFgEta(int fgEta)
{
  m_fgEta = fgEta;
}

void HGCalCluster::setFgPhi(int fgPhi)
{
  m_fgPhi = fgPhi;
}

void HGCalCluster::setHOverE(int hOverE)
{
  m_hOverE = hOverE;
}

void HGCalCluster::setFgECAL(int fgECAL)
{
  m_fgECAL = fgECAL;
}

bool HGCalCluster::checkClusterFlag(ClusterFlag flag) const 
{
  return (m_clusterFlags & (0x1<<flag));
};

bool HGCalCluster::isValid() const
{
    return ( checkClusterFlag(INCLUDE_SEED) );
}

int HGCalCluster::hwPtEm()const
{
  return m_hwPtEm;
}

int HGCalCluster::hwPtHad()const
{
  return m_hwPtHad;
}

int HGCalCluster::hwSeedPt() const
{
  return m_hwSeedPt;
}

int HGCalCluster::fgEta() const
{
  return m_fgEta;
}

int HGCalCluster::fgPhi() const
{
  return m_fgPhi;
}

int HGCalCluster::hOverE() const
{
  return m_hOverE;
}

int HGCalCluster::fgECAL() const
{
  return m_fgECAL;
}

bool HGCalCluster::operator<(const HGCalCluster& cl) const
{
  bool res = false;
  // Favour high pT
  if(hwPt()<cl.hwPt()) res = true;
  else if(hwPt()==cl.hwPt()) {
    // Favour central clusters
    if( abs(hwEta())>abs(cl.hwEta()) ) res = true;
    else if( abs(hwEta())==abs(cl.hwEta()) ){
      // Favour small phi (arbitrary)
      if(hwPhi()>cl.hwPhi()) res = true;
    }
  }
  return res;
}

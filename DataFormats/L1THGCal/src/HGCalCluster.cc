#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;

HGCalCluster::HGCalCluster( const LorentzVector p4, 
			     int pt,
			     int eta,
			     int phi)
  : L1Candidate(p4, pt, eta, phi)
{
  
}

HGCalCluster::~HGCalCluster() 
{
  
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

#include "DataFormats/L1THGCal/interface/HGCalCluster3D.h"

using namespace l1t;

HGCalCluster3D::HGCalCluster3D( const LorentzVector p4, 
			     int pt,
			     int eta,
			     int phi)
  : L1Candidate(p4, pt, eta, phi)
{
  
}

HGCalCluster3D::~HGCalCluster3D() 
{
  
}

bool HGCalCluster3D::operator<(const HGCalCluster3D& cl) const
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

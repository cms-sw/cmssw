#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

using namespace l1t;


HGCalMulticluster::HGCalMulticluster( const LorentzVector p4, 
                                      int pt,
                                      int eta,
                                      int phi,
                                      ClusterCollection &basic_clusters
    ) :
    L1Candidate(p4, pt, eta, phi),
    myclusters_(basic_clusters)
{
} 


HGCalMulticluster::HGCalMulticluster( const l1t::HGCalCluster & clu )
{

    centre_ = clu.centreNorm();
    hwPt_ = clu.hwPt();
    mipPt_ = clu.mipPt();
    zside_ = clu.zside();

}


HGCalMulticluster::~HGCalMulticluster() 
{
  
}


bool HGCalMulticluster::isPertinent( const l1t::HGCalCluster & clu, double dR ) const
{
    
    if( zside_ != clu.zside() )
        return false;

    if( ( centre_ - clu.centreNorm() ).Mag2() < dR )
        return true;
    
    return false;

}


void HGCalMulticluster::addClu( const l1t::HGCalCluster & clu ) const
{

    centre_ = ( centre_*mipPt_ + clu.centreNorm()*clu.mipPt() ) / ( mipPt_+clu.mipPt() ) ;
    
    mipPt_ = mipPt_ + clu.mipPt();
    hwPt_ = hwPt_ + clu.hwPt();

}


bool HGCalMulticluster::operator<(const HGCalMulticluster& cl) const
{

  bool res = false;
  // Favour high pT
  if( mipPt() < cl.mipPt() ) 
      res = true;
  else if( mipPt() == cl.mipPt() ) {
    // Favour central clusters
    if( abs(hwEta()) > abs(cl.hwEta()) ) 
        res = true;
    else if( abs(hwEta())==abs(cl.hwEta()) )
      // Favour small phi (arbitrary)
      if(hwPhi()>cl.hwPhi()) 
          res = true; 
  }

  return res;

}

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

using namespace l1t;

HGCalMulticluster::HGCalMulticluster( const l1t::HGCalCluster & clu )
{

    centreNorm_ = clu.centreNorm();
    centre_ = clu.centre();
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

    if( ( centreNorm_ - clu.centreNorm() ).Mag2() < dR )
        return true;
    
    return false;

}

void HGCalMulticluster::addClu( const l1t::HGCalCluster & clu )
{

    /* update c3d position */
    centre_ =  centre_*mipPt_ + clu.centre()*clu.mipPt();
    centre_ = centre_ / ( mipPt_+clu.mipPt() ) ;
 
    centreNorm_ =  centreNorm_*mipPt_ + clu.centreNorm()*clu.mipPt();
    centreNorm_ = centreNorm_ / ( mipPt_+clu.mipPt() ) ;
        
    /* update c3d energies */
    mipPt_ += clu.mipPt();
    hwPt_ += clu.hwPt();
 
    math::PtEtaPhiMLorentzVector p4( this->p4() );
    p4 += clu.p4();
    this->setP4( p4 );

    clusters_.push_back(0, &clu );
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

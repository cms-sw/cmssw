#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

using namespace l1t;

HGCalMulticluster::HGCalMulticluster( const l1t::HGCalCluster & clu )
    : firstClusterDetId_( clu.seedDetId() )
{

    centreNorm_ = clu.centreNorm();
    centre_ = clu.centre();
    mipPt_ = clu.mipPt();
}

HGCalMulticluster::HGCalMulticluster( const LorentzVector p4, 
                                      int pt,
                                      int eta,
                                      int phi
    ) :
    L1Candidate(p4, pt, eta, phi){
    } 

HGCalMulticluster::~HGCalMulticluster() 
{
  
}


bool HGCalMulticluster::isPertinent( const l1t::HGCalCluster & clu, double dR ) const
{
    HGCalDetId cluDetId( clu.seedDetId() );
    HGCalDetId firstClusterDetId( firstClusterDetId_ );
    
    if( cluDetId.zside() != firstClusterDetId.zside() ){
        return false;
    }
    if( ( centreNorm_ - clu.centreNorm() ).mag() < dR ){
        return true;
    }
    return false;

}

void HGCalMulticluster::addCluster( const l1t::HGCalCluster & clu )
{

    if( clusters_.size() == 0 ){ 
        firstClusterDetId_ = clu.seedDetId();
    }


    /* update c3d position */
    centre_ =  centre_*mipPt_ + clu.centre()*clu.mipPt();
    centre_ = centre_ / ( mipPt_+clu.mipPt() ) ;
 
    centreNorm_ =  centreNorm_*mipPt_ + clu.centreNorm()*clu.mipPt();
    centreNorm_ = centreNorm_ / ( mipPt_+clu.mipPt() ) ;
        
    /* update c3d energies */
    mipPt_ += clu.mipPt();
  
    int hwPt = 0;
    hwPt += clu.hwPt();
    this->setHwPt(hwPt);
 
    math::PtEtaPhiMLorentzVector p4( this->p4() );
    p4 += clu.p4();
    this->setP4( p4 );

    clusters_.push_back(0, &clu );
}

int32_t HGCalMulticluster::zside() const
{
    
    HGCalDetId firstClusterDetId( firstClusterDetId_ );    
    
    return firstClusterDetId.zside();

}


bool HGCalMulticluster::operator<(const HGCalMulticluster& cl) const
{

  bool res = false;
  // Favour high pT
  if( mipPt() < cl.mipPt() ) 
      res = true;
//  else if( mipPt() == cl.mipPt() ) {
//    // Favour central clusters
//    if( abs(hwEta()) > abs(cl.hwEta()) ) 
//        res = true;
//    else if( abs(hwEta())==abs(cl.hwEta()) )
//      // Favour small phi (arbitrary)
//      if(hwPhi()>cl.hwPhi()) 
//          res = true; 
//  }

  return res;

}

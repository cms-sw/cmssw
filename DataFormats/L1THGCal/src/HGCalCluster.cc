#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

using namespace l1t;

HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi
    ): 
    L1Candidate(p4, pt, eta, phi),
    centre_(0, 0, 0),
    mipPt_(0),
    seedMipPt_(0)
{

}




HGCalCluster::HGCalCluster(  const l1t::HGCalTriggerCell &tcSeed )
    : seedDetId_( tcSeed.detId() ),
      centre_(0, 0, 0),
      mipPt_(0),
      seedMipPt_(0)
{
    addTriggerCell( tcSeed );
}



HGCalCluster::~HGCalCluster()
{

}



bool HGCalCluster::isPertinent( const l1t::HGCalTriggerCell &tc, double distEtaPhi ) const 
{

    HGCalDetId tcDetId( tc.detId() );
    HGCalDetId seedDetId( seedDetId_ );
    if( tcDetId.layer() != seedDetId.layer() ||
        tcDetId.subdetId() != seedDetId.subdetId() ||
        tcDetId.zside() != seedDetId.zside() )
        return false;
   
    if ( this->distance(tc) < distEtaPhi )
        return true;

    return false;

}


void HGCalCluster::addTriggerCell(const l1t::HGCalTriggerCell &tc)
{

    if( tcs_.size() == 0 ){ 
        seedDetId_ = tc.detId();
        seedMipPt_ = tc.mipPt();
    }

    GlobalVector tcXYZ( tc.position().x(),
                             tc.position().y(),
                             tc.position().z() );
    /* update c2d positions */
    centre_ = centre_*mipPt_ + tcXYZ*tc.mipPt() ;
    centre_ = centre_ / ( mipPt_ + tc.mipPt() );

    /* update c2d energies */
    mipPt_ += tc.mipPt();

    int hwPt = 0;
    hwPt  += tc.hwPt();
    this->setHwPt(hwPt);

    math::PtEtaPhiMLorentzVector p4( ( this->p4() )  );
    p4 += tc.p4(); 
    this->setP4( p4 );

    tcs_.push_back(0, &tc );


}


double HGCalCluster::distance(const l1t::HGCalTriggerCell &tc) const
{

    GlobalVector tcPointXYZ( tc.position().x(), 
                             tc.position().y(), 
                             tc.position().z() );
    
    return ( tcPointXYZ - centre_ ).mag();
   
}


uint32_t HGCalCluster::subdetId()  const
{

    HGCalDetId seedDetId( seedDetId_ );
    
    return seedDetId.subdetId();

}


uint32_t HGCalCluster::layer() const
{
    
    HGCalDetId seedDetId( seedDetId_ );    
    
    return seedDetId.layer();

}


int32_t HGCalCluster::zside() const
{
    
    HGCalDetId seedDetId( seedDetId_ );    
    
    return seedDetId.zside();

}


bool HGCalCluster::operator<(const HGCalCluster& cl) const
{

    bool res = false;

    /* Prioratize high pT */
    if( mipPt() < cl.mipPt()) {
        res = true;
    }
    //else if( mipPt() == cl.mipPt() ) {
    //    if( abs(hwEta()) > abs( cl.hwEta() ) ) /* Prioratize central clusters */
    //        res = true;
    //    else if( abs(hwEta())==abs( cl.hwEta() ) )
    //        if( hwPhi() > cl.hwPhi())         /* Prioratize small phi (arbitrary) */
    //            res = true;
    //}

    return res;

}


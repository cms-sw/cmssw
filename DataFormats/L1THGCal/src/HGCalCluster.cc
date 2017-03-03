#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;


//HGCalCluster::HGCalCluster( const LorentzVector p4, 
//                            int pt,
//                            int eta,
//                            int phi,
//                            tc_collection &basic_tc
//    ): 
//    L1Candidate(p4, pt, eta, phi),
//    centre_(0, 0, 0),
//    hwPt_(0),
//    mipPt_(0)
//{
//
//}

HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi
    ): 
    L1Candidate(p4, pt, eta, phi),
    centre_(0, 0, 0),
    hwPt_(0),
    mipPt_(0)
{

}




HGCalCluster::HGCalCluster(  const l1t::HGCalTriggerCell &tcSeed )
    : seedDetId_( tcSeed.detId() ),
      centre_(0, 0, 0),
      hwPt_(0),
      mipPt_(0)
{
    addTC( tcSeed );
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
   
    if ( this->dist(tc) < distEtaPhi )
        return true;

    return false;

}


void HGCalCluster::addTC(const l1t::HGCalTriggerCell &tc)
{

    if( tcs_.size() == 0 ) 
        seedDetId_ = tc.detId();

    ROOT::Math::XYZVector tcPointXYZ( tc.position().x(), 
                                      tc.position().y(), 
                                      tc.position().z() );

    /* update c2d positions */
    centre_ = centre_*mipPt_ + tcPointXYZ*tc.mipPt() ;
    centre_ = centre_ / ( mipPt_ + tc.mipPt() );

    /* update c2d energies */
    mipPt_ += tc.mipPt();
    hwPt_  += tc.hwPt();

    math::PtEtaPhiMLorentzVector p4( ( this->p4() )  );
    p4 += tc.p4(); 
    this->setP4( p4 );

    tcs_.push_back(0, &tc );


}


double HGCalCluster::dist(const l1t::HGCalTriggerCell &tc) const
{

    ROOT::Math::XYZVector tcPointXYZ( tc.position().x(), 
                                      tc.position().y(), 
                                      tc.position().z() );

    return TMath::Sqrt( ( tcPointXYZ - centre_ ).Mag2() );
   
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
    else if( mipPt() == cl.mipPt() ) {
        if( abs(hwEta()) > abs( cl.hwEta() ) ) /* Prioratize central clusters */
            res = true;
        else if( abs(hwEta())==abs( cl.hwEta() ) )
            if( hwPhi() > cl.hwPhi())         /* Prioratize small phi (arbitrary) */
                res = true;
    }

    return res;

}


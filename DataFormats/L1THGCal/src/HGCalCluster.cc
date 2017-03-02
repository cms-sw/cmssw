#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;


HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi)
    : L1Candidate(p4, pt, eta, phi), 
      centre_(0, 0, 0),
      hwPt_(0),
      mipPt_(0)
{

}


HGCalCluster::HGCalCluster(  const l1t::HGCalTriggerCell &tcSeed,
                             //edm::PtrVector<l1t::HGCalTriggerCell> tcCollection,
                             const edm::EventSetup & es,
                             const edm::Event & evt )  
    : seedDetId_( tcSeed.detId() ),
      centre_(0, 0, 0),
//      tcPtrs_( tcCollection ),
      hwPt_(0),
      mipPt_(0)
{
    recHitTools_.getEvent( evt );
    recHitTools_.getEventSetup( es );
    addTCseed( tcSeed );
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


void HGCalCluster::addTC(const l1t::HGCalTriggerCell &tc) const
{

    DetId id( tc.detId() );
    ROOT::Math::RhoEtaPhiVector tcPoint( recHitTools_.getPosition( id ).z(), 
                                         tc.eta(), 
                                         tc.phi() );

    ROOT::Math::XYZVector tcPointXYZ( tcPoint.X(), 
                                      tcPoint.Y(), 
                                      tcPoint.Z() );

    centre_ = centre_*mipPt_ + tcPointXYZ*tc.mipPt() ;
    centre_ = centre_ / ( mipPt_ + tc.mipPt() );

    mipPt_ = mipPt_ + tc.mipPt();
    hwPt_  = hwPt_ + tc.hwPt();
  
}


void HGCalCluster::addTCseed(const l1t::HGCalTriggerCell &tc) const
{
    
    seedDetId_ = tc.detId();
    DetId id( seedDetId_ );
    ROOT::Math::RhoEtaPhiVector tcPoint( recHitTools_.getPosition( id ).z(), 
                                         tc.eta(), 
                                         tc.phi() );

    centre_.SetXYZ( tcPoint.X(), 
                    tcPoint.Y(), 
                    tcPoint.Z() );

    mipPt_ = tc.mipPt();
    hwPt_ = tc.hwPt();

}


double HGCalCluster::dist(const l1t::HGCalTriggerCell &tc) const
{

    DetId id( tc.detId() );
    ROOT::Math::RhoEtaPhiVector tcPoint( recHitTools_.getPosition( id ).z(), 
                                         tc.eta(), 
                                         tc.phi() );
    
    ROOT::Math::XYZVector tcPointXYZ( tcPoint.X(), 
                                      tcPoint.Y(), 
                                      tcPoint.Z() );

    return ( tcPointXYZ - centre_ ).Mag2();
    

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


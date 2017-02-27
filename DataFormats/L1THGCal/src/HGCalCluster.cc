#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;


HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi)
    : L1Candidate(p4, pt, eta, phi), 
      centre_(0, 0, 0),
      hwPt_(0)
{

}


HGCalCluster::HGCalCluster(  const l1t::HGCalTriggerCell &tcSeed )  
    : seedDetId_( tcSeed.detId() ),
      centre_(0, 0, 0),
      hwPt_(0)
{
    addTCseed( tcSeed );
}



HGCalCluster::~HGCalCluster()
{

}



bool HGCalCluster::isPertinent( const l1t::HGCalTriggerCell &tc, double distEtaPhi ) const 
{

    HGCalDetId tcDetId( tc.detId() );
    HGCalDetId seedDetId( seedDetId_ );
    if( tcDetId.layer() != seedDetId.layer() && 
        tcDetId.subdetId() != seedDetId.subdetId() )
        return false;

    ROOT::Math::RhoEtaPhiVector tcPoint( 0, tc.eta(), tc.phi() );

    double dist =  ( tcPoint - centre_ ).Mag2();

    if ( dist < distEtaPhi )
        return true;

    return false;

}


void HGCalCluster::addTC(const l1t::HGCalTriggerCell &tc) const
{

    ROOT::Math::RhoEtaPhiVector tcPoint( 0, tc.eta(), tc.phi() );

    centre_ = centre_*hwPt_ + tcPoint*tc.hwPt() ;
    centre_ = centre_ / ( hwPt_ + tc.hwPt() );

    hwPt_ = hwPt_ + tc.hwPt();

}


void HGCalCluster::addTCseed(const l1t::HGCalTriggerCell &tc) const
{

    ROOT::Math::RhoEtaPhiVector tcPoint( 0, tc.eta(), tc.phi() );

    centre_ = tcPoint ;

    hwPt_   = tc.hwPt();

    seedDetId_ = tc.detId();

}


double HGCalCluster::dist(const l1t::HGCalTriggerCell &tc) const
{

    return reco::deltaR ( tc.p4().Eta(), tc.p4().Phi(), 
                          this->p4().Eta(), this->p4().Phi() );

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


bool HGCalCluster::operator<(const HGCalCluster& cl) const
{

    bool res = false;

    /* Prioratize high pT */
    if(hwPt()<cl.hwPt()) {
        res = true;
    }
    else if(hwPt()==cl.hwPt()) {
        if( abs(hwEta()) > abs( cl.hwEta() ) ) /* Prioratize central clusters */
            res = true;
        else if( abs(hwEta())==abs( cl.hwEta() ) )
            if( hwPhi() > cl.hwPhi())         /* Prioratize small phi (arbitrary) */
                res = true;
    }

    return res;

}


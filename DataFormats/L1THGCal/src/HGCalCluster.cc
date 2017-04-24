#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

using namespace l1t;

HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi )
   : L1Candidate(p4, pt, eta, phi),
     seedDetId_(0),
     centre_(0, 0, 0),
     centreProj_(0., 0., 0.),
     mipPt_(0),
     seedMipPt_(0)
{

}


HGCalCluster::HGCalCluster( const edm::Ptr<l1t::HGCalTriggerCell> &tcSeed )
    : seedDetId_( tcSeed->detId() ),
      centre_(0., 0., 0.),
      centreProj_(0., 0., 0.),
      mipPt_(0.),
      seedMipPt_(0.)
{
    addTriggerCell( tcSeed );
}


HGCalCluster::~HGCalCluster()
{

}


void HGCalCluster::addTriggerCell( const edm::Ptr<l1t::HGCalTriggerCell > &tc )
{
    
    if( triggercells_.empty() ){ 
        seedDetId_ = tc->detId();
        seedMipPt_ = tc->mipPt();
    }

    /* update cluster positions */
    Basic3DVector<float> tcVector( tc->position() );
    Basic3DVector<float> centreVector( centre_ );

    centreVector = centreVector*mipPt_ + tcVector*tc->mipPt();
    if( mipPt_ + tc->mipPt()!=0 ){
        centreVector = centreVector / ( mipPt_ + tc->mipPt() ) ;
    }
    
    centre_ = GlobalPoint( centreVector );
    
    if( centreVector.z()!=0 ){
        centreProj_= GlobalPoint( centreVector / centreVector.z() );
    }
    /* update cluster energies */
    mipPt_ += tc->mipPt();

    int hwPt = this->hwPt() + tc->hwPt();
    this->setHwPt(hwPt);

    math::PtEtaPhiMLorentzVector p4( ( this->p4() )  );
    p4 += tc->p4(); 
    this->setP4( p4 );

    triggercells_.push_back( tc );

}


double HGCalCluster::distance(const l1t::HGCalTriggerCell &tc) const
{

    return ( tc.position() - centre_ ).mag();
   
}


uint32_t HGCalCluster::subdetId()  const
{

    return this->hgcalSeedDetId().subdetId();

}


uint32_t HGCalCluster::layer() const
{
    
    return this->hgcalSeedDetId().layer();

}


int32_t HGCalCluster::zside() const
{

    return this->hgcalSeedDetId().zside();

}


bool HGCalCluster::operator<(const HGCalCluster& cl) const
{

    /* Prioratize high pT */
    return (mipPt() < cl.mipPt());

}

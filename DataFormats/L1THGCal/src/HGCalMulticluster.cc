#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

using namespace l1t;

HGCalMulticluster::HGCalMulticluster( const l1t::HGCalCluster & clu )
    : firstClusterDetId_( clu.seedDetId() )
{

    centreProj_ = clu.centreProj();
    centre_ = clu.centre();
    mipPt_ = clu.mipPt();
}


HGCalMulticluster::HGCalMulticluster( const LorentzVector p4, 
                                      int pt,
                                      int eta,
                                      int phi ) 
    : L1Candidate(p4, pt, eta, phi)
{

} 


HGCalMulticluster::~HGCalMulticluster() 
{
  
}


void HGCalMulticluster::addCluster( const l1t::HGCalCluster & clu )
{

    if( clusters_.size() == 0 ){ 
        firstClusterDetId_ = clu.seedDetId();
    }
    
    /* update multicluster positions */
    Basic3DVector<float> centreVector( centre_.x(),
                                       centre_.y(), 
                                       centre_.z() );

    Basic3DVector<float> cluCentreVector( clu.centre().x(), 
                                          clu.centre().y(), 
                                          clu.centre().z() );

    centreVector = centreVector*mipPt_ + cluCentreVector*clu.mipPt();
    centreVector = centreVector / ( mipPt_+clu.mipPt() ) ;

    GlobalPoint centreAux((float)centreVector.x(), 
                          (float)centreVector.y(), 
                          (float)centreVector.z() );
    centre_ = centreAux;

    Basic3DVector<float> centreProjVector( centreProj_.x(), 
                                           centreProj_.y(),
                                           centreProj_.z() );

    Basic3DVector<float> cluCentreProjVector( clu.centreProj().x(), 
                                              clu.centreProj().y(), 
                                              clu.centreProj().z() );

    centreProjVector = centreProjVector*mipPt_ + cluCentreProjVector*clu.mipPt();
    centreProjVector = centreProjVector / ( mipPt_+clu.mipPt() ) ;

    GlobalPoint centreProjAux( (float)centreProjVector.x(), 
                               (float)centreProjVector.y(), 
                               (float)centreProjVector.z() );

    centreProj_ = centreProjAux;
        
    /* update multicluster energies */
    mipPt_ += clu.mipPt();
  
    int hwPt = 0;
    hwPt += clu.hwPt();
    this->setHwPt(hwPt);
 
    math::PtEtaPhiMLorentzVector p4( this->p4() );
    p4 += clu.p4();
    this->setP4( p4 );

}

void HGCalMulticluster::addClusterList( edm::Ptr<l1t::HGCalCluster> &clu)
{

    clusters_.push_back( clu );

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

  return res;

}

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

using namespace l1t;

HGCalMulticluster::HGCalMulticluster( const LorentzVector p4, 
                                      int pt,
                                      int eta,
                                      int phi ) 
    : L1Candidate(p4, pt, eta, phi),
      firstClusterDetId_(0),
      centre_(0,0,0),
      centreProj_(0,0,0),
      mipPt_(0)
{

} 


HGCalMulticluster::HGCalMulticluster( const edm::Ptr<l1t::HGCalCluster>& clu )
    : firstClusterDetId_( clu->seedDetId() ),
      centre_( clu->centre() ),
      centreProj_( clu->centreProj() ),
      mipPt_( clu->mipPt() )
{
    addCluster( clu );
}


HGCalMulticluster::~HGCalMulticluster() 
{
  
}


void HGCalMulticluster::addCluster( const edm::Ptr<l1t::HGCalCluster> & clu )
{

    if( clusters_.size() == 0 ){ 
        firstClusterDetId_ = clu->seedDetId();
    }
    
    /* update multicluster positions */
    Basic3DVector<float> centreVector( centre_ );
    Basic3DVector<float> cluCentreVector( clu->centre() );

    centreVector = centreVector*mipPt_ + cluCentreVector*clu->mipPt();
    centreVector = centreVector / ( mipPt_+clu->mipPt() ) ;

    centre_ = GlobalPoint( centreVector );

    Basic3DVector<float> centreProjVector( centreProj_ );
    Basic3DVector<float> cluCentreProjVector( clu->centreProj() );

    centreProjVector = centreProjVector*mipPt_ + cluCentreProjVector*clu->mipPt();
    centreProjVector = centreProjVector / ( mipPt_+clu->mipPt() ) ;

    centreProj_ = GlobalPoint( centreProjVector );
        
    /* update multicluster energies */
    mipPt_ += clu->mipPt();
  
    int hwPt = this->hwPt() + clu->hwPt();
    this->setHwPt(hwPt);
 
    math::PtEtaPhiMLorentzVector p4( this->p4() );
    p4 += clu->p4();
    this->setP4( p4 );

    clusters_.push_back( clu );

}


int32_t HGCalMulticluster::zside() const
{
    
    HGCalDetId firstClusterDetId( firstClusterDetId_ );    
    
    return firstClusterDetId.zside();

}

double HGCalMulticluster::hOverE() const
{

    double pt_em = 0.;
    double pt_had = 0.;
    double hOe = 0.;

    const edm::PtrVector<l1t::HGCalCluster>& cls = this->clusters();
    for( edm::PtrVector<l1t::HGCalCluster>::iterator iclu = cls.begin(); iclu!=cls.end(); ++iclu){
        if( (*iclu)->subdetId() == 3 ){
            pt_em += (*iclu)->p4().Pt();
        }
        else if( (*iclu)->subdetId() > 3 ){
            pt_had += (*iclu)->p4().Pt();
        }        
    }
    if(pt_em>0) hOe = pt_had / pt_em ;
    else hOe = -1.;

    return hOe;

}


bool HGCalMulticluster::operator<(const HGCalMulticluster& cl) const
{

    /* Prioratize high pT */
    return (mipPt() < cl.mipPt());

}

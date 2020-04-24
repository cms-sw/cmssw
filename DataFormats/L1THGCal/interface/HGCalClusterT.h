#ifndef DataFormats_L1Trigger_HGCalClusterT_h
#define DataFormats_L1Trigger_HGCalClusterT_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include "Math/Vector3D.h"


namespace l1t 
{
  template <class C> class HGCalClusterT : public L1Candidate 
  {

    public:
      typedef typename edm::PtrVector<C>::const_iterator const_iterator;

    public:
      HGCalClusterT(){}
      HGCalClusterT( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          )
        : L1Candidate(p4, pt, eta, phi),
        valid_(true),
        detId_(0),
        centre_(0, 0, 0),
        centreProj_(0., 0., 0.),
        mipPt_(0),
        seedMipPt_(0){}

      HGCalClusterT( const edm::Ptr<C>& c ):
        valid_(true),
        detId_( c->detId() ),
        centre_(0., 0., 0.),
        centreProj_(0., 0., 0.),
        mipPt_(0.),
        seedMipPt_(0.)
      {
        addConstituent(c);
      }
      
      ~HGCalClusterT() override {};
      
      const edm::PtrVector<C>& constituents() const {return constituents_;}        
      const_iterator constituents_begin() const {return constituents_.begin();}
      const_iterator constituents_end() const {return constituents_.end();}
      unsigned size() const { return constituents_.size(); }

      void addConstituent( const edm::Ptr<C>& c )
      {
        if( constituents_.empty() )
        { 
          detId_ = HGCalDetId(c->detId());
          seedMipPt_ = c->mipPt();
        }

        /* update cluster positions */
        Basic3DVector<float> constituentCentre( c->position() );
        Basic3DVector<float> clusterCentre( centre_ );

        clusterCentre = clusterCentre*mipPt_ + constituentCentre*c->mipPt();
        if( mipPt_ + c->mipPt()!=0 ) 
        {
          clusterCentre /= ( mipPt_ + c->mipPt() ) ;
        }
        centre_ = GlobalPoint( clusterCentre );

        if( clusterCentre.z()!=0 ) 
        {
          centreProj_= GlobalPoint( clusterCentre / clusterCentre.z() );
        }
        /* update cluster energies */
        mipPt_ += c->mipPt();

        int updatedPt = hwPt() + c->hwPt();
        setHwPt(updatedPt);

        math::PtEtaPhiMLorentzVector updatedP4 ( p4() );
        updatedP4 += c->p4(); 
        setP4( updatedP4 );

        constituents_.push_back( c );

      }
      
      bool valid() const { return valid_;}
      void setValid(bool valid) { valid_ = valid;}
      
      double mipPt() const { return mipPt_; }
      double seedMipPt() const { return seedMipPt_; }
      uint32_t detId() const { return detId_.rawId(); }

      /* distance in 'cm' */
      double distance( const l1t::HGCalTriggerCell &tc ) const 
      {
        return ( tc.position() - centre_ ).mag();
      }

      const GlobalPoint& position() const { return centre_; } 
      const GlobalPoint& centre() const { return centre_; }
      const GlobalPoint& centreProj() const { return centreProj_; }

      // FIXME: will need to fix places where the shapes are directly accessed
      // Right now keep shapes() getter as non-const 
      ClusterShapes& shapes() {return shapes_;}
      double hOverE() const
      {
        double pt_em = 0.;
        double pt_had = 0.;
        double hOe = 0.;

        for(const auto& constituent : constituents())
        {
          switch( constituent->subdetId() )
          {
            case HGCEE:
              pt_em += constituent->pt();
              break;
            case HGCHEF:
              pt_had += constituent->pt();
              break;
            case HGCHEB:
              pt_had += constituent->pt();
              break;
            default:
              break;
          }
        }
        if(pt_em>0) hOe = pt_had / pt_em ;
        else hOe = -1.;
        return hOe;
      }

      uint32_t subdetId() const {return detId_.subdetId();} 
      uint32_t layer() const {return detId_.layer();}
      int32_t zside() const {return detId_.zside();}
      

      /* operators */
      bool operator<(const HGCalClusterT<C>& cl) const {return mipPt() < cl.mipPt();}
      bool operator>(const HGCalClusterT<C>& cl) const  { return  cl<*this;   }
      bool operator<=(const HGCalClusterT<C>& cl) const { return !(cl>*this); }
      bool operator>=(const HGCalClusterT<C>& cl) const { return !(cl<*this); }

    private:
        
      bool valid_;
      HGCalDetId detId_;     
      edm::PtrVector<C> constituents_;
      GlobalPoint centre_;
      GlobalPoint centreProj_; // centre projected onto the first HGCal layer

      double mipPt_;
      double seedMipPt_;

      ClusterShapes shapes_;

  };

}

#endif

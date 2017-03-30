#ifndef DataFormats_L1Trigger_HGCalCluster_h
#define DataFormats_L1Trigger_HGCalCluster_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include "Math/Vector3D.h"


namespace l1t {
    
  class HGCalCluster : public L1Candidate {
    
    public:
       
      /* constructors and destructor */
      HGCalCluster(){}
      HGCalCluster( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          );

      HGCalCluster( const edm::Ptr<l1t::HGCalTriggerCell> &tc );
      
      ~HGCalCluster();
      
      /* trigger-cells collection pertinent to the cluster */
      const edm::PtrVector<l1t::HGCalTriggerCell>& triggercells() const {
          return triggercells_; 
      }        
      edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator triggercells_begin() const { 
          return triggercells_.begin(); 
      }
      edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator triggercells_end() const { 
          return triggercells_.end(); 
      }
      const edm::Ptr<l1t::HGCalTriggerCell> seedTriggerCell() const {
          return *triggercells_begin(); 
      }
      unsigned triggercellsSize() const { return triggercells_.size(); }

      /* helpers */
      void addTriggerCell( const edm::Ptr<l1t::HGCalTriggerCell> &tc);
      
      /* set info */
      void setModule  (uint32_t value) { module_   = value; }
      
      /* get info */
      bool isValid() const { 
          if(triggercells_.size() > 0 ) return true;  
          else return false;
      }
      double mipPt()       const { return mipPt_; }
      double seedMipPt()   const { return seedMipPt_; }
      uint32_t seedDetId() const { return seedDetId_; }

      bool containsSeed(double seedThr_MipT){ 
          for(edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggercells_.begin(); tc != triggercells_.end(); ++tc){
              if( (*tc)->mipPt() > seedThr_MipT ){
                  return true;
              }
          }    
          return false;
      }
      
      double distance( const l1t::HGCalTriggerCell &tc ) const; /* return distance in 'cm' */
      
      const GlobalPoint& centre() const { return centre_; }
      const GlobalPoint& centreProj() const { return centreProj_; }

      uint32_t subdetId()  const; /* EE (3), FH (4) or BH (5) */
      uint32_t layer()     const;
      int32_t zside()      const;
      uint32_t module()    const { return module_; }

      ClusterShapes shapes;

      /* operators */
      bool operator<(const HGCalCluster& cl) const;
      bool operator>(const HGCalCluster& cl) const  { return  cl<*this;   }
      bool operator<=(const HGCalCluster& cl) const { return !(cl>*this); }
      bool operator>=(const HGCalCluster& cl) const { return !(cl<*this); }

    private:
        
      /* persistent vector of edm::Ptr to trigger-cells that build up the cluster */
      edm::PtrVector<l1t::HGCalTriggerCell> triggercells_;
        
      /* seed detId */
      uint32_t seedDetId_;
 
      /* Centre weighted with energy */
      GlobalPoint centre_;

      /* Centre weighted with energy */
      GlobalPoint centreProj_;

      /* Energies */
      double mipPt_;
      double seedMipPt_;

      /* HGC specific information */
      uint32_t module_;
         
  };

  typedef BXVector<HGCalCluster> HGCalClusterBxCollection;

}

#endif

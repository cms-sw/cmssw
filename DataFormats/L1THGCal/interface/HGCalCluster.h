#ifndef DataFormats_L1Trigger_HGCalCluster_h
#define DataFormats_L1Trigger_HGCalCluster_h

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
                      int pt,
                      int eta,
                      int phi
            );
       
        HGCalCluster( const l1t::HGCalTriggerCell &tc ); 

        ~HGCalCluster();

        /* trigger-cell collection pertinent to the cluster */
        BXVector<const l1t::HGCalTriggerCell*>  tcs() const { return tcs_; }        

        /* helpers */
        void addTriggerCell( const l1t::HGCalTriggerCell &tc );

        /* set info */
        void setModule  (uint32_t value) { module_   = value; }

        /* get info */
        bool isValid()      const { return true;  }
        double mipPt()      const { return mipPt_; }
        double seedMipPt() const { return seedMipPt_; }
        uint32_t seedDetId() const { return seedDetId_; }

        double distance( const l1t::HGCalTriggerCell &tc ) const; /* return distance in 'cm' */
        
        GlobalVector centre() const { return centre_; }
        GlobalVector centreNorm() const { return centre_/centre_.z(); }

        uint32_t subdetId()  const; /* EE (3), FH (4) or BH (5) */
        uint32_t layer()     const;
        int32_t zside()     const;
        uint32_t module()    const { return module_; }

        ClusterShapes shapes;

        /* operations */
        bool operator<(const HGCalCluster& cl) const;
        bool operator>(const HGCalCluster& cl) const  { return  cl<*this;   }
        bool operator<=(const HGCalCluster& cl) const { return !(cl>*this); }
        bool operator>=(const HGCalCluster& cl) const { return !(cl<*this); }

    private:
        
        
        BXVector<const l1t::HGCalTriggerCell*> tcs_;
          
        /* seed detId */
        uint32_t seedDetId_;
 
        /* Centre weighted with energy */
        GlobalVector centre_;

        /* Energies */
        double mipPt_;
        double seedMipPt_;

        /* HGC specific information */
        uint32_t module_;
         
        /* identification variables */
        uint32_t hOverE_; 
         
    };

    typedef BXVector<HGCalCluster> HGCalClusterBxCollection;

}

#endif

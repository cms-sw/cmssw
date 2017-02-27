
#ifndef DataFormats_L1Trigger_HGCalCluster_h
#define DataFormats_L1Trigger_HGCalCluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "Math/Vector3D.h"

/* temporary */
#include "DataFormats/L1THGCal/interface/GeoTmp.h"

namespace l1t {

    class HGCalCluster : public L1Candidate {
    public:

        /* constructors and destructor */
        HGCalCluster(){}
        HGCalCluster( const LorentzVector p4,
                      int pt=0,
                      int eta=0,
                      int phi=0 );
        HGCalCluster( const l1t::HGCalTriggerCell &tc );
        
        ~HGCalCluster();

        /* helpers */
        bool isPertinent( const l1t::HGCalTriggerCell &tc, double dR ) const;
        void addTC( const l1t::HGCalTriggerCell &tc ) const;
        void addTCseed( const l1t::HGCalTriggerCell &tc ) const;

        /* set info */
        void setModule  (uint32_t value) { module_   = value; }

        /* get info */
        bool isValid()      const { return true;  }
        uint32_t hwPt()     const { return hwPt_; }
        ROOT::Math::RhoEtaPhiVector centre() const { return centre_; }
        ROOT::Math::RhoEtaPhiVector centreNorm() const;
        //uint32_t hwSeedPt() const { return hwSeedPt_; }
        double dist( const l1t::HGCalTriggerCell &tc ) const;
        
        uint32_t subdetId()  const; /* EE (3), FH (4) or BH (5) */
        uint32_t layer()     const;
        int32_t zside()     const;
        uint32_t module()    const { return module_; }

        ClusterShapes shapes; /* ??? */

        bool operator<(const HGCalCluster& cl) const;
        bool operator>(const HGCalCluster& cl) const  { return  cl<*this;   }
        bool operator<=(const HGCalCluster& cl) const { return !(cl>*this); }
        bool operator>=(const HGCalCluster& cl) const { return !(cl<*this); }
        //bool operator+(const HGCalCluster& cl) const; /* to be implemented */
        //bool operator-(const HGCalCluster& cl) const; /* to be implemented */


    private:
        
        /* seed detId */
        mutable uint32_t seedDetId_;

        /* Centre weighted with energy */
        mutable ROOT::Math::RhoEtaPhiVector centre_;

        /* Energies */
        mutable uint32_t hwPt_;
        uint32_t hwSeedPt_;

        /* HGC specific information */
        uint32_t module_;

        /* identification variables */
        uint32_t hOverE_; 

        /* private methods */
        //double dist_( const l1t::HGCalTriggerCell &tc ) const;

    };

    typedef BXVector<HGCalCluster> HGCalClusterBxCollection;

}

#endif

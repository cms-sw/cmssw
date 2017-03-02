#ifndef DataFormats_L1Trigger_HGCalCluster_h
#define DataFormats_L1Trigger_HGCalCluster_h

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/Common/interface/PtrVector.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Math/Vector3D.h"


namespace l1t {

    class HGCalCluster : public L1Candidate {
    public:
        
        typedef edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc_iterator;
        typedef edm::PtrVector<l1t::HGCalTriggerCell> tc_collection;

        /* constructors and destructor */
        HGCalCluster(){}
        HGCalCluster( const LorentzVector p4,
                      int pt,
                      int eta,
                      int phi,
                      tc_collection &thecls
        );
       
        HGCalCluster( const LorentzVector p4,
                      int pt,
                      int eta,
                      int phi
        );


        HGCalCluster( const l1t::HGCalTriggerCell &tc, 
                      //edm::PtrVector<l1t::HGCalTriggerCell> tcCollection,
                      const edm::EventSetup & es,
                      const edm::Event & evt );
        
        ~HGCalCluster();

        /* trigger-cell collection pertinent to the cluster*/
        const edm::PtrVector<l1t::HGCalTriggerCell> & tcs() const { return tcs_; }        
        unsigned int size() const { return tcs_.size(); }  
        tc_iterator begin() const { return tcs_.begin(); }
        tc_iterator end() const { return tcs_.end(); }

        /* helpers */
        bool isPertinent( const l1t::HGCalTriggerCell &tc, double dR ) const;
        void addTC( const l1t::HGCalTriggerCell &tc );
        void addTCseed( const l1t::HGCalTriggerCell &tc );

        /* set info */
        void setModule  (uint32_t value) { module_   = value; }

        /* get info */
        bool isValid()      const { return true;  }
        uint32_t hwPt()     const { return hwPt_; }
        double mipPt()      const { return mipPt_; }
        //uint32_t hwSeedPt() const { return hwSeedPt_; }
        double dist( const l1t::HGCalTriggerCell &tc ) const; /* return distance in 'cm' */
        
        ROOT::Math::XYZVector centre() const { return centre_; }
        ROOT::Math::XYZVector centreNorm() const { return centre_/centre_.z(); }

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
        
        
        edm::PtrVector<l1t::HGCalTriggerCell>  tcs_;
        
        /* tools for geometry */
        hgcal::RecHitTools recHitTools_;
         
        /* seed detId */
        uint32_t seedDetId_;
 
        /* Centre weighted with energy */
        ROOT::Math::XYZVector centre_;

        /* Energies */
        uint32_t hwPt_;
        double mipPt_;
        uint32_t hwSeedPt_;

        /* HGC specific information */
        uint32_t module_;
         
        /* identification variables */
        uint32_t hOverE_; 
         
    };

    typedef BXVector<HGCalCluster> HGCalClusterBxCollection;

}

#endif

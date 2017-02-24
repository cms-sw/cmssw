
#ifndef DataFormats_L1Trigger_HGCalCluster_h
#define DataFormats_L1Trigger_HGCalCluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/ClusterShapes.h"

#include "DataFormats/Math/interface/deltaPhi.h"


namespace l1t {

    class HGCalCluster : public L1Candidate {
    public:

        HGCalCluster(){}
        
        HGCalCluster( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          );

        HGCalCluster(const l1t::HGCalTriggerCell &tc) {
            //this->addTC( tc );
        }
        
        ~HGCalCluster();

        bool isPertinent( const l1t::HGCalTriggerCell &tc, double dist_eta ) const;
        
        void addTC(const l1t::HGCalTriggerCell &tc);
        double dist(const l1t::HGCalTriggerCell &tc);

        void setHwPtEm  (uint32_t value) { hwPtEm_   = value; }
        void setHwPtHad (uint32_t value) { hwPtHad_  = value; }
        void setHwSeedPt(uint32_t value) { hwSeedPt_ = value; }
        void setSubDet  (uint32_t value) { subDet_   = value; }
        void setLayer   (uint32_t value) { layer_    = value; }
        void setModule  (uint32_t value) { module_   = value; }
        void setHOverE  (uint32_t value) { hOverE_   = value; }

        bool isValid()      const { return true;      }
        uint32_t hwPtEm()   const { return hwPtEm_;   } 
        uint32_t hwPtHad()  const { return hwPtHad_;  }
        uint32_t hwSeedPt() const { return hwSeedPt_; }

        uint32_t subDet()  const { return subDet_; }
        uint32_t layer()   const { return layer_;  }
        uint32_t module()  const { return module_; }

        uint32_t hOverE() const { return hOverE_; }

        ClusterShapes shapes ;

        bool operator<(const HGCalCluster& cl) const;
        bool operator>(const HGCalCluster& cl) const  { return  cl<*this;   }
        bool operator<=(const HGCalCluster& cl) const { return !(cl>*this); }
        bool operator>=(const HGCalCluster& cl) const { return !(cl<*this); }

    private:
        
        /* Centre weighted with energy */
        LorentzVector centre_; 

        // Energies
        uint32_t hwPtEm_;
        uint32_t hwPtHad_;
        uint32_t hwSeedPt_;

        // HGC specific information
        uint32_t subDet_;
        uint32_t layer_;
        uint32_t module_;

        // identification variables
        uint32_t hOverE_; 
    };

    typedef BXVector<HGCalCluster> HGCalClusterBxCollection;

}

#endif

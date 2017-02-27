#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"


namespace l1t {
  
  class HGCalMulticluster : public L1Candidate {
    public:

        /* constructors and destructor */
        HGCalMulticluster() {}
        HGCalMulticluster( const LorentzVector p4,
                           int pt,
                           int eta,
                           int phi,
                           ClusterCollection &basic_clusters
            );
        HGCalMulticluster( const l1t::HGCalCluster & clu );

        ~HGCalMulticluster();
        void push_back(const edm::Ptr<l1t::HGCalCluster> &b) {
            myclusters_.push_back(b);
        }
  
        const edm::PtrVector<l1t::HGCalCluster> & clusters() const { return myclusters_; }
        
        unsigned int size() const { return myclusters_.size(); }  
        component_iterator begin() const { return myclusters_.begin(); }
        component_iterator end() const { return myclusters_.end(); }

        void push_back(const edm::Ptr<l1t::HGCalCluster> &b){ 
            clusters_.push_back(b);
        }
        
//        void setHwPtEm  (uint32_t pt)    {hwPtEm_= pt;}
//        void setHwPtHad (uint32_t pt)    {hwPtHad_ = pt;}
        void setHwSeedPt(uint32_t pt)    {hwSeedPt_ = pt;}
        void setSubDet  (uint32_t subdet){subDet_ = subdet;}
        void setNtotLayer   (uint32_t nTotLayer) {nTotLayer_ = nTotLayer;}
        void setHOverE  (uint32_t hOverE){hOverE_ = hOverE;}
        
        /* helpers */
        bool isPertinent( const l1t::HGCalCluster & clu, double dR ) const;
        void addClu( const l1t::HGCalCluster & clu ) const;
        //void addCluSeed( const l1t::HGCalCluster & clu ) const;
        /* set info */

        /* get info */
        bool isValid()      const {return true;}
        ROOT::Math::RhoEtaPhiVector centre() const { return centre_; }
//uint32_t hwPtEm()   const {return hwPtEm_;}
        //uint32_t hwPtHad()  const {return hwPtHad_;}
//        uint32_t hwSeedPt() const {return hwSeedPt_;}
        
//        uint32_t subDet() const {return subDet_;}
        uint32_t nTotLayer()  const {return nTotLayer_;}
        
        uint32_t hOverE() const { return hOverE_; }
        
        bool operator<(const HGCalMulticluster& cl) const;
        bool operator>(const HGCalMulticluster& cl) const {return  cl<*this;};
        bool operator<=(const HGCalMulticluster& cl) const {return !(cl>*this);};
        bool operator>=(const HGCalMulticluster& cl) const {return !(cl<*this);};
        
    private:
        edm::PtrVector<l1t::HGCalCluster>  myclusters_;

        // Energies
//        uint32_t hwPtEm_;
//        uint32_t hwPtHad_;
        uint32_t hwSeedPt_;

        /* Energies */
        mutable uint32_t hwPt_;

        /* centre in norm plane */
        mutable ROOT::Math::RhoEtaPhiVector centre_;
        // HGC specific information
        /* detId of the first cluster in the multicluster */
        uint32_t firstCluId_;

        uint32_t nTotLayer_;
        
        // identification variables
        uint32_t hOverE_; 

        int32_t zside_;
    
    };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;
  
  
}

#endif

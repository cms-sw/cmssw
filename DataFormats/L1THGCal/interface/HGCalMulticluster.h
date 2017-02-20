#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

namespace l1t {
  
  class HGCalMulticluster : public L1Candidate {
    public:
        
        typedef edm::PtrVector<l1t::HGCalCluster>::const_iterator component_iterator;
        typedef edm::PtrVector<l1t::HGCalCluster>  ClusterCollection;

        HGCalMulticluster(){}
        HGCalMulticluster( const LorentzVector p4,
                           int pt,
                           int eta,
                           int phi,
                           ClusterCollection &thecls
            );

        ~HGCalMulticluster();
        
        void push_back(const edm::Ptr<l1t::HGCalCluster> &b) {
            myclusters.push_back(b);
        }
  
        const edm::PtrVector<l1t::HGCalCluster> & clusters() const { return myclusters; }
        
        unsigned int size() const { return myclusters.size(); }  
        component_iterator begin() const { return myclusters.begin(); }
        component_iterator end() const { return myclusters.end(); }
        
        void setHwPtEm  (uint32_t pt)    {hwPtEm_= pt;}
        void setHwPtHad (uint32_t pt)    {hwPtHad_ = pt;}
        void setHwSeedPt(uint32_t pt)    {hwSeedPt_ = pt;}
        void setSubDet  (uint32_t subdet){subDet_ = subdet;}
        void setNtotLayer   (uint32_t nTotLayer) {nTotLayer_ = nTotLayer;}
        void setHOverE  (uint32_t hOverE){hOverE_ = hOverE;}
        
        bool isValid()      const {return true;}
        uint32_t hwPtEm()   const {return hwPtEm_;}
        uint32_t hwPtHad()  const {return hwPtHad_;}
        uint32_t hwSeedPt() const {return hwSeedPt_;}
        
        uint32_t subDet() const {return subDet_;}
        uint32_t nTotLayer()  const {return nTotLayer_;}
        
        uint32_t hOverE() const {return hOverE_;}
        
        bool operator<(const HGCalMulticluster& cl) const;
        bool operator>(const HGCalMulticluster& cl) const {return  cl<*this;};
        bool operator<=(const HGCalMulticluster& cl) const {return !(cl>*this);};
        bool operator>=(const HGCalMulticluster& cl) const {return !(cl<*this);};
        
    private:
        edm::PtrVector<l1t::HGCalCluster>  myclusters;

        // Energies
        uint32_t hwPtEm_;
        uint32_t hwPtHad_;
        uint32_t hwSeedPt_;
        
        // HGC specific information
        uint32_t subDet_;
        uint32_t nTotLayer_;
        
        // identification variables
        uint32_t hOverE_; 
    };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;
  
  
}

#endif

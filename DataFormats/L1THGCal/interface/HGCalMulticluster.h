#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/Common/interface/PtrVector.h"

namespace l1t {
  
  class HGCalMulticluster : public L1Candidate {
    public:
        typedef edm::PtrVector<l1t::HGCalCluster>::const_iterator clu_iterator;
        typedef edm::PtrVector<l1t::HGCalCluster> clu_collection;
        
        /* constructors and destructor */
        HGCalMulticluster() {}
        HGCalMulticluster( const LorentzVector p4,
                           int pt,
                           int eta,
                           int phi,
                           clu_collection &thecls
        );
        
        HGCalMulticluster( const l1t::HGCalCluster & clu );

        ~HGCalMulticluster();

        void push_back(const edm::Ptr<l1t::HGCalCluster> &b) {
            clusters_.push_back(b);
        }

        /* cluster collection pertinent to the multicluster*/
        const edm::PtrVector<l1t::HGCalCluster> & clusters() const { return clusters_; }        
        unsigned int size() const { return clusters_.size(); }  
        clu_iterator begin() const { return clusters_.begin(); }
        clu_iterator end() const { return clusters_.end(); }
        

        /* helpers */
        bool isPertinent( const l1t::HGCalCluster & clu, double dR ) const;
        void addClu( const l1t::HGCalCluster & clu );

        /* get info */

        bool isValid()      const {return true;}
        ROOT::Math::XYZVector centre() const { return centre_; } /* in normalized plane (x/z, y/z, z/z)*/

        uint32_t hwPt() const { return hwPt_; }
        double mipPt() const { return mipPt_; }

        uint32_t nTotLayer()  const { return nTotLayer_; } /* not working */
        uint32_t hOverE() const { return hOverE_; } /* not working */
        
        bool operator<(const HGCalMulticluster& cl) const;
        bool operator>(const HGCalMulticluster& cl) const {return  cl<*this;};
        bool operator<=(const HGCalMulticluster& cl) const {return !(cl>*this);};
        bool operator>=(const HGCalMulticluster& cl) const {return !(cl<*this);};
        
    private:


        edm::PtrVector<l1t::HGCalCluster>  clusters_;

        /* Energies */
        uint32_t hwPt_;
        double mipPt_;

        /* centre in norm plane */
        ROOT::Math::XYZVector centre_;

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

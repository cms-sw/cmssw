#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

namespace l1t {
  
  class HGCalMulticluster : public L1Candidate {

    public:

        /* constructors and destructor */
        HGCalMulticluster() {}

        HGCalMulticluster( const LorentzVector p4,
                           int pt=0,
                           int eta=0,
                           int phi=0
            );

        HGCalMulticluster( const edm::Ptr<l1t::HGCalCluster> &clu );
 
        ~HGCalMulticluster();

        /* cluster collection pertinent to the multicluster */
        const edm::PtrVector<l1t::HGCalCluster>&  clusters() const {
            return clusters_; 
        }        
        edm::PtrVector<l1t::HGCalCluster>::const_iterator clusters_begin() const { 
            return clusters_.begin(); 
        }
        edm::PtrVector<l1t::HGCalCluster>::const_iterator clusters_end() const { 
            return clusters_.end(); 
        }
        const edm::Ptr<l1t::HGCalCluster> firstCluster() const {
            return *clusters_begin(); 
        }        
        unsigned clustersSize() const { return clusters_.size(); }
        
        /* helpers */
        void addCluster( const edm::Ptr<l1t::HGCalCluster> &clu);

        /* get info */
        bool isValid() const {
            if(clusters_.size() > 0 ) return true;
            return false;
        }

        const GlobalPoint& centre() const { return centre_; }         /* in (x, y, z) */
        const GlobalPoint& centreProj() const { return centreProj_; } /* in (x/z, y/z, z/z) */

        uint32_t firstClusterDetId() const { return firstClusterDetId_; }
        double mipPt() const { return mipPt_; }
        double hOverE() const;
        int32_t zside() const;

        /* operators */
        bool operator<(const HGCalMulticluster& cl) const;
        bool operator>(const HGCalMulticluster& cl) const {return  cl<*this;};
        bool operator<=(const HGCalMulticluster& cl) const {return !(cl>*this);};
        bool operator>=(const HGCalMulticluster& cl) const {return !(cl<*this);};
        
    private:

        /* persistent vector of edm::Ptr to clusters that build up the multicluster */
        edm::PtrVector<l1t::HGCalCluster> clusters_;

        /* detId of the first cluster in the multicluster */
        uint32_t firstClusterDetId_;

        /* centre in norm plane */
        GlobalPoint centre_;

        /* barycentre */
        GlobalPoint centreProj_;

        /* Energies */
        double mipPt_;
            
    };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;  
  
}

#endif

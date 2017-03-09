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
                           int pt,
                           int eta,
                           int phi
        );

        HGCalMulticluster( const l1t::HGCalCluster & clu );

        ~HGCalMulticluster();

        /* cluster collection pertinent to the multicluster */
        const edm::PtrVector<l1t::HGCalCluster>  clusters() const { return clusters_; }        

        /* helpers */
        void addCluster( const l1t::HGCalCluster & clu );
        void addClusterList( edm::Ptr<l1t::HGCalCluster> &clu );

        /* get info */
        bool isValid()      const {return true;}
        GlobalPoint centre() const { return centre_; } /* in normal plane (x, y, z) */
        GlobalPoint centreProj() const { return centreProj_; } /* in normalized plane (x/z, y/z, z/z) */

        uint32_t firstClusterDetId() const { return firstClusterDetId_; }
        double mipPt() const { return mipPt_; }
        uint32_t hOverE() const { return hOverE_; }
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
      
        /* identification variables */
        uint32_t hOverE_;
      
    };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;  
  
}

#endif

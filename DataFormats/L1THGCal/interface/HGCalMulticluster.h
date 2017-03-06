#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/Common/interface/PtrVector.h"


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

        /* cluster collection pertinent to the multicluster*/
        BXVector<const l1t::HGCalCluster*> clusters() const { return clusters_; }        

        /* helpers */
        bool isPertinent( const l1t::HGCalCluster & clu, double dR ) const;
        void addCluster( const l1t::HGCalCluster & clu );

        /* get info */
        bool isValid()      const {return true;}
//        ROOT::Math::XYZVector centre() const { return centre_; } /* in normal plane (x, y, z)*/
//        ROOT::Math::XYZVector centreNorm() const { return centreNorm_; } /* in normalized plane (x/z, y/z, z/z)*/
        GlobalVector centre() const { return centre_; } /* in normal plane (x, y, z)*/
        GlobalVector centreNorm() const { return centreNorm_; } /* in normalized plane (x/z, y/z, z/z)*/

        uint32_t firstClusterDetId() const { return firstClusterDetId_; }
        double mipPt() const { return mipPt_; }
        uint32_t hOverE() const { return hOverE_; }
        int32_t zside() const;
             
        bool operator<(const HGCalMulticluster& cl) const;
        bool operator>(const HGCalMulticluster& cl) const {return  cl<*this;};
        bool operator<=(const HGCalMulticluster& cl) const {return !(cl>*this);};
        bool operator>=(const HGCalMulticluster& cl) const {return !(cl<*this);};
        
    private:

        BXVector<const l1t::HGCalCluster*> clusters_;

        /* detId of the first cluster in the multicluster */
        uint32_t firstClusterDetId_;
        /* centre in norm plane */
        //ROOT::Math::XYZVector centre_;
        GlobalVector centre_;

        /* barycentre */
        //ROOT::Math::XYZVector centreNorm_;
        GlobalVector centreNorm_;

        /* Energies */
        double mipPt_;
      
        // HGC specific information
        
        // identification variables
        uint32_t hOverE_;
      
    };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;
  
  
}

#endif

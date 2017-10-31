#ifndef DataFormats_L1Trigger_HGCalMulticluster_h
#define DataFormats_L1Trigger_HGCalMulticluster_h

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
#include "DataFormats/L1THGCal/interface/HGCalClusterT.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

namespace l1t {
            
  class HGCalMulticluster : public HGCalClusterT<l1t::HGCalCluster> {
    
    public:
       
      HGCalMulticluster(){}
      HGCalMulticluster( const LorentzVector p4,
          int pt=0,
          int eta=0,
          int phi=0
          );

      HGCalMulticluster( const edm::Ptr<l1t::HGCalCluster> &tc );
      
      ~HGCalMulticluster() override;


  };
    
  typedef BXVector<HGCalMulticluster> HGCalMulticlusterBxCollection;  
  
}

#endif

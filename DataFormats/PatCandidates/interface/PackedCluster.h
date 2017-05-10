#ifndef __DataFormats_PatCandidates_PackedCluster_h__
#define __DataFormats_PatCandidates_PackedCluster_h__


//This class is designed to record basic information (et,eta,phi and seed detId about a CaloCluster
//it has to be extremely lightweight for space reasons

#include "DataFormats/DetId/interface/DetId.h"

namespace reco {
  class CaloCluster;
}

namespace pat {
  class PackedCluster {
    
  public:
    PackedCluster():
      packedEt_(0),packedEta_(0),packedPhi_(0),
      seedId_(0),
      unpacked_(false),
      unpackedEt_(0),unpackedEta_(0),unpackedPhi_(0){}
    
    PackedCluster(const reco::CaloCluster& clus);

    void pack(bool unpackAfterwards=true);
    void unpack()const;

    float et()const{if(!unpacked_) unpack(); return unpackedEt_;}
    float eta()const{if(!unpacked_) unpack(); return unpackedEta_;}
    float phi()const{if(!unpacked_) unpack(); return unpackedPhi_;}
    DetId seedId()const{return seedId_;}

  private:
    uint16_t packedEt_, packedEta_,packedPhi_;
    DetId seedId_;
    mutable bool unpacked_;
    mutable float unpackedEt_, unpackedEta_,unpackedPhi_;

   
  };
}

#endif

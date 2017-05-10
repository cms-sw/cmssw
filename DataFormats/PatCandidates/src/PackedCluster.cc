
#include "DataFormats/PatCandidates/interface/PackedCluster.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"

pat::PackedCluster::PackedCluster(const reco::CaloCluster& clus):
  packedEt_(0),packedEta_(0),packedPhi_(0),
  seedId_(clus.seed()),
  unpacked_(true),
  unpackedEt_(sin(clus.position().theta())*clus.energy()),
  unpackedEta_(clus.eta()),
  unpackedPhi_(clus.phi())
{
  pack();
}

void pat::PackedCluster::pack(bool unpackAfterwards)
{  
  packedEt_  =  MiniFloatConverter::float32to16(unpackedEt_);
  packedEta_ =  int16_t(std::round(unpackedEta_/6.0f*std::numeric_limits<int16_t>::max()));
  packedPhi_ =  int16_t(std::round(unpackedPhi_/3.2f*std::numeric_limits<int16_t>::max()));
  if(unpackAfterwards) unpack();
}

void pat::PackedCluster::unpack()const
{
  unpackedEt_ = MiniFloatConverter::float16to32(packedEt_);
  unpackedEta_ = int16_t(packedEta_)*6.0f/std::numeric_limits<int16_t>::max();
  unpackedPhi_ = int16_t(packedPhi_)*3.2f/std::numeric_limits<int16_t>::max();
  unpacked_=true;
  
}

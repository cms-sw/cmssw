#ifndef ECALEBCLUSTERTRIGGERPRIMITIVEDIGI_H
#define ECALEBCLUSTERTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalEBClusterTriggerPrimitiveSample.h"



/** \class EcalEBClusterTriggerPrimitiveDigi
\author N. Marinelli - Univ. of Notre Dame


*/

class EcalEBClusterTriggerPrimitiveDigi {
 public:
  typedef EBDetId key_type; ///< For the sorted collection

  EcalEBClusterTriggerPrimitiveDigi(); // for persistence
  EcalEBClusterTriggerPrimitiveDigi(const EBDetId& tpId, const std::vector< EBDetId>& xtalIDs, float etaClu, float phiClu);
  

  void swap(EcalEBClusterTriggerPrimitiveDigi& rh) {
    std::swap(tpId_,rh.tpId_);
    std::swap(size_,rh.size_);
    std::swap(data_,rh.data_);
  }
  
  const EBDetId& id() const { return tpId_; }
  int size() const { return size_; }
    
  const EcalEBClusterTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const EcalEBClusterTriggerPrimitiveSample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  void setSample(int i, const EcalEBClusterTriggerPrimitiveSample& sam);
  void setSampleValue(int i, uint32_t value) { data_[i].setValue(value); }
    
  static const int MAXSAMPLES = 20;

  /// get the 10 bits Et of interesting sample
  int encodedEt() const; 

  /// get the EBDetId of crystals used in the cluster
  std::vector<EBDetId> crystalsInCluster() const {return cryIdInCluster_;}

  // get energy  weighted eta and phi of the cluster
  float eta() const  {return etaClu_;}
  float phi() const  {return phiClu_;}

  /// Spike flag
  int l1aSpike() const;

  /// Time info
  int time() const;

  /// True if debug mode (# of samples > 1)
  bool isDebug() const;

  /// Gets the interesting sample
  int sampleOfInterest() const;

private:
 
  EBDetId tpId_;
  std::vector<EBDetId>  cryIdInCluster_;
  float etaClu_;
  float phiClu_;
  int size_;
  std::vector<EcalEBClusterTriggerPrimitiveSample> data_;


};


inline void swap(EcalEBClusterTriggerPrimitiveDigi& lh, EcalEBClusterTriggerPrimitiveDigi& rh) {
  lh.swap(rh);
}

std::ostream& operator<<(std::ostream& s, const EcalEBClusterTriggerPrimitiveDigi& digi);



#endif

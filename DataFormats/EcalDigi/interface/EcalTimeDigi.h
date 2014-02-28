#ifndef ECALTIMEDIGI_H
#define ECALTIMEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

/** \class EcalTimeDigi

$Id : $
*/

class EcalTimeDigi {
 public:
  typedef DetId key_type; ///< For the sorted collection

  EcalTimeDigi(); // for persistence
  explicit EcalTimeDigi(const DetId& id);
  

  void swap(EcalTimeDigi& rh) {
    std::swap(id_,rh.id_);
    std::swap(size_,rh.size_);
    std::swap(data_,rh.data_);
  }
  
  const DetId& id() const { return id_; }
  int size() const { return size_; }
    
  const float& operator[](unsigned int i) const { return data_[i]; }
  const float& sample(unsigned int i) const { return data_[i]; }
    
  void setSize(unsigned int size);
  void setSample(unsigned int i, const float& sam) { data_[i]=sam; }
  void setSampleOfInterest(int i) { sampleOfInterest_=i; }
    
  static const unsigned int MAXSAMPLES = 10;

  /// Gets the BX==0 sample. If =-1 then it means that only OOT hits are present
  int sampleOfInterest() const { return sampleOfInterest_; }

private:
  
  DetId id_;
  unsigned int size_;
  int sampleOfInterest_;
  std::vector<float> data_;
};


inline void swap(EcalTimeDigi& lh, EcalTimeDigi& rh) {
  lh.swap(rh);
}

std::ostream& operator<<(std::ostream& s, const EcalTimeDigi& digi);



#endif

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
    
  const float& operator[](int i) const { return data_[i]; }
  const float& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  void setSample(int i, const float& sam) { data_[i]=sam; }
    
  static const int MAXSAMPLES = 10;

  /// Gets the interesting sample
  int sampleOfInterest() const;

private:
  
  DetId id_;
  int size_;
  std::vector<float> data_;
};


inline void swap(EcalTimeDigi& lh, EcalTimeDigi& rh) {
  lh.swap(rh);
}

std::ostream& operator<<(std::ostream& s, const EcalTimeDigi& digi);



#endif

#ifndef ECALTRIGGERPRIMITIVEDIGI_H
#define ECALTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"



/** \class EcalTriggerPrimitiveDigi
      
$Id : $
*/

class EcalTriggerPrimitiveDigi {
 public:
  typedef EcalTrigTowerDetId key_type; ///< For the sorted collection

  EcalTriggerPrimitiveDigi(); // for persistence
  explicit EcalTriggerPrimitiveDigi(const EcalTrigTowerDetId& id);
    
  const EcalTrigTowerDetId& id() const { return id_; }
  int size() const { return size_; }
    
  const EcalTriggerPrimitiveSample& operator[](const int& i) const { return data_[i]; }
  const EcalTriggerPrimitiveSample& sample(const int& i) const { return data_[i]; }
    
  void setSize(const int& size);
  void setSample(const int& i, const EcalTriggerPrimitiveSample& sam) { data_[i]=sam; }
  void setSampleValue(const int& i, const uint16_t& value) { data_[i].setValue(value); }
    
  static const int MAXSAMPLES = 20;

  /// get the encoded/compressed Et of interesting sample
  int compressedEt() const; 
  
  
  /// get the fine-grain bit of interesting sample
  bool fineGrain() const; 
  
  /// get the Trigger tower Flag of interesting sample
  int ttFlag() const; 
  
  /// True if debug mode (# of samples > 1)
  bool isDebug() const;

  /// Gets the interesting sample
  int sampleOfInterest() const;
  
 private:
  
  EcalTrigTowerDetId id_;
  int size_;
  std::vector<EcalTriggerPrimitiveSample> data_;
};


std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveDigi& digi);



#endif

#ifndef ECALTRIGGERPRIMITIVEDIGI_H
#define ECALTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"



/** \class EcalTriggerPrimitiveDigi

see also EcalTrigPrimCompactColl.

*/

class EcalTriggerPrimitiveDigi {
 public:
  //typedef EcalTrigTowerDetId key_type; ///< For the sorted collection
  typedef EBDetId key_type; ///< For the sorted collection

  EcalTriggerPrimitiveDigi(); // for persistence
  // explicit EcalTriggerPrimitiveDigi(const EcalTrigTowerDetId& id);
  EcalTriggerPrimitiveDigi(const EBDetId& id);
  

  void swap(EcalTriggerPrimitiveDigi& rh) {
    std::swap(id_,rh.id_);
    std::swap(size_,rh.size_);
    std::swap(data_,rh.data_);
  }
  
  const EBDetId& id() const { return id_; }
  //const EcalTrigTowerDetId& id() const { return id_; }
  int size() const { return size_; }
    
  const EcalTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const EcalTriggerPrimitiveSample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  // void setSample(int i, const EcalTriggerPrimitiveSample& sam) {cout << " Fuck you " << endl; data_[i]=sam; }
  void setSample(int i, const EcalTriggerPrimitiveSample& sam);
  void setSampleValue(int i, uint16_t value) { data_[i].setValue(value); }
    
  static const int MAXSAMPLES = 20;

  /// get the encoded/compressed Et of interesting sample
  int compressedEt() const; 
  
  
  /// get the fine-grain bit of interesting sample
  bool fineGrain() const; 
  
  /// get the Trigger tower Flag of interesting sample
  int ttFlag() const; 

  /// Gets the "strip fine grain veto bit" (sFGVB) used as L1A spike detection
  /// @return 0 spike like pattern
  ///         1 EM shower like pattern
  int sFGVB() const;

  /// Gets the L1A spike detection flag. Beware the flag is inverted.
  /// Deprecated, use instead sFGVB() method, whose name is less missleading
  /// @return 0 spike like pattern
  ///         1 EM shower like pattern
  int l1aSpike() const { return sFGVB(); }
  
  /// True if debug mode (# of samples > 1)
  bool isDebug() const;

  /// Gets the interesting sample
  int sampleOfInterest() const;

private:
  EBDetId id_;
  //EcalTrigTowerDetId id_;
  int size_;
  std::vector<EcalTriggerPrimitiveSample> data_;
};


inline void swap(EcalTriggerPrimitiveDigi& lh, EcalTriggerPrimitiveDigi& rh) {
  lh.swap(rh);
}

std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveDigi& digi);



#endif

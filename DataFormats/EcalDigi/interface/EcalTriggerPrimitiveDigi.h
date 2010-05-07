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
    
  const EcalTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const EcalTriggerPrimitiveSample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  void setSample(int i, const EcalTriggerPrimitiveSample& sam) { data_[i]=sam; }
  void setSampleValue(int i, uint16_t value) { data_[i].setValue(value); }
    
  static const int MAXSAMPLES = 20;

  /// get the encoded/compressed Et of interesting sample
  int compressedEt() const; 
  
  
  /// get the fine-grain bit of interesting sample
  bool fineGrain() const; 
  
  /// get the Trigger tower Flag of interesting sample
  int ttFlag() const; 

  /// gets the L1A spike detection flag. 
  /// @return 1 if the trigger primitive was forced to zero because a spike was detected by L1 trigger,
  ///         0 if it wasn't
  ///         -1 if failed to retrieve the sample of interest (see #sampleOfInterest())), that contains this information:
  int l1aSpike() const;
  
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

#ifndef HCALTRIGGERPRIMITIVEDIGI_H
#define HCALTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

/** \class HcalTriggerPrimitiveDigi
    
$Date: 2005/12/10 16:44:38 $
$Revision: 1.6 $
\author J. Mans - Minnesota
*/
class HcalTriggerPrimitiveDigi {
public:
  typedef HcalTrigTowerDetId key_type; ///< For the sorted collection

  HcalTriggerPrimitiveDigi(); // for persistence
  explicit HcalTriggerPrimitiveDigi(const HcalTrigTowerDetId& id);
  
  const HcalTrigTowerDetId& id() const { return id_; }
  int size() const { return size_; }
  int presamples() const { return hcalPresamples_; }
  
  const HcalTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const HcalTriggerPrimitiveSample& sample(int i) const { return data_[i]; }

  /// Full "Sample of Interest"
  const HcalTriggerPrimitiveSample& t0() const { return data_[hcalPresamples_]; }  
  /// Fine-grain bit for the "Sample of Interest"
  bool SOI_fineGrain() const { return t0().fineGrain(); }
  /// Compressed ET for the "Sample of Interest"
  bool SOI_compressedEt() const { return t0().compressedEt(); }

  void setSize(int size);
  void setPresamples(int ps);
  void setSample(int i, const HcalTriggerPrimitiveSample& sam) { data_[i]=sam; }
  
  static const int MAXSAMPLES = 10;
private:
  HcalTrigTowerDetId id_;
  int size_;
  int hcalPresamples_;
  HcalTriggerPrimitiveSample data_[MAXSAMPLES];
};

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveDigi& digi);


#endif

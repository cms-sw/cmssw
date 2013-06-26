#ifndef HCALTRIGGERPRIMITIVEDIGI_H
#define HCALTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

/** \class HcalTriggerPrimitiveDigi
    
$Date: 2009/06/09 13:15:56 $
$Revision: 1.10 $
\author J. Mans - Minnesota
*/
class HcalTriggerPrimitiveDigi {
public:
  typedef HcalTrigTowerDetId key_type; ///< For the sorted collection

  HcalTriggerPrimitiveDigi(); // for persistence
  explicit HcalTriggerPrimitiveDigi(const HcalTrigTowerDetId& id);
  
  const HcalTrigTowerDetId& id() const { return id_; }
  int size() const { return (size_&0xF); }
  int presamples() const { return hcalPresamples_&0xF; }

   /// was ZS MarkAndPass?
  bool zsMarkAndPass() const { return (hcalPresamples_&0x10); }
  /// was ZS unsuppressed?
  bool zsUnsuppressed() const { return (hcalPresamples_&0x20); }

  void setZSInfo(bool unsuppressed, bool markAndPass);

  const HcalTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
  const HcalTriggerPrimitiveSample& sample(int i) const { return data_[i]; }

  /// Full "Sample of Interest"
  const HcalTriggerPrimitiveSample& t0() const { return data_[presamples()]; }  
  /// Fine-grain bit for the "Sample of Interest"
  bool SOI_fineGrain() const { return t0().fineGrain(); }
  /// Compressed ET for the "Sample of Interest"
  int SOI_compressedEt() const { return t0().compressedEt(); }

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

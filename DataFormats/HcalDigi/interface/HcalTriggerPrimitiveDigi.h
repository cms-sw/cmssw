#ifndef HCALTRIGGERPRIMITIVEDIGI_H
#define HCALTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "DataFormats/HcalDigi/interface/HcalTriggerPrimitiveSample.h"

/** \class HcalTriggerPrimitiveDigi
    
$Date: 2005/10/04 13:37:35 $
$Revision: 1.4 $
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
  const HcalTriggerPrimitiveSample& t0() const { return data_[hcalPresamples_]; }
  
  void setSize(int size);
  void setPresamples(int ps);
  void setSample(int i, const HcalTriggerPrimitiveSample& sam) { data_[i]=sam; }
  
  static const int MAXSAMPLES = 10;
private:
  HcalTrigTowerDetId id_;
  int size_;
  int hcalPresamples_;
  std::vector<HcalTriggerPrimitiveSample> data_;
};

std::ostream& operator<<(std::ostream& s, const HcalTriggerPrimitiveDigi& digi);


#endif

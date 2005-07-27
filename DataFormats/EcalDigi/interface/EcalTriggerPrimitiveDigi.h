#ifndef ECALTRIGGERPRIMITIVEDIGI_H
#define ECALTRIGGERPRIMITIVEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"

namespace cms {

  /** \class EcalTriggerPrimitiveDigi
      
  $Id : $
  */

  class EcalTriggerPrimitiveDigi {
  public:
    EcalTriggerPrimitiveDigi(); // for persistence
    explicit EcalTriggerPrimitiveDigi(const EcalTrigTowerDetId& id);
    
    const EcalTrigTowerDetId& id() const { return id_; }
    int size() const { return size_; }
    
    const EcalTriggerPrimitiveSample& operator[](int i) const { return data_[i]; }
    const EcalTriggerPrimitiveSample& sample(int i) const { return data_[i]; }
    
    void setSize(int size);
    void setSample(int i, const EcalTriggerPrimitiveSample& sam) { data_[i]=sam; }
    
    static const int MAXSAMPLES = 8;
  private:
    EcalTrigTowerDetId id_;
    int size_;
    std::vector<EcalTriggerPrimitiveSample> data_;
  };


  std::ostream& operator<<(std::ostream& s, const EcalTriggerPrimitiveDigi& digi);

}



#endif

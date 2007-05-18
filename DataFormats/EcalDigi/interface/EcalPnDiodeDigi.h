#ifndef ECALPNDIODEDIGI_H
#define ECALPNDIODEDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"
#include "DataFormats/EcalDigi/interface/EcalFEMSample.h"



/** \class EcalPnDiodeDigi
      
$Id: EcalPnDiodeDigi.h,v 1.2 2007/03/27 09:55:01 meridian Exp $
*/

class EcalPnDiodeDigi {
 public:
  typedef EcalPnDiodeDetId key_type; ///< For the sorted collection

  EcalPnDiodeDigi(); // for persistence
  explicit EcalPnDiodeDigi(const EcalPnDiodeDetId& id);
    
  const EcalPnDiodeDetId& id() const { return id_; }
  int size() const { return size_; }
    
  const EcalFEMSample& operator[](int i) const { return data_[i]; }
  const EcalFEMSample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  void setSample(int i, const EcalFEMSample& sam) { data_[i]=sam; }
    
  static const int MAXSAMPLES = 50;
 private:
  EcalPnDiodeDetId id_;
  int size_;
  std::vector<EcalFEMSample> data_;
};


std::ostream& operator<<(std::ostream& s, const EcalPnDiodeDigi& digi);



#endif

#ifndef ECALMATACQDIGI_H
#define ECALMATACQDIGI_H 1

#include <ostream>
#include <vector>
#include "DataFormats/EcalDigi/interface/EcalMatacqSample.h"



/** \class EcalMatacqDigi
      
$Id: EcalMatacqDigi.h,v 1.1 2005/10/11 07:47:04 meridian Exp $
*/

class EcalMatacqDigi {
 public:
  EcalMatacqDigi(); // for persistence
    
  int size() const { return size_; }
    
  const EcalMatacqSample& operator[](int i) const { return data_[i]; }
  const EcalMatacqSample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  void setSample(int i, const EcalMatacqSample& sam) { data_[i]=sam; }
    
  static const int MAXSAMPLES = 2000;

 private:

  int size_;
  std::vector<EcalMatacqSample> data_;
};


std::ostream& operator<<(std::ostream& s, const EcalMatacqDigi& digi);



#endif

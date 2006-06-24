#ifndef DIGIECAL_ECALDATAFRAME_H
#define DIGIECAL_ECALDATAFRAME_H

#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include <ostream>



/** \class EcalDataFrame
      
$Id: $
*/

class EcalDataFrame {
 public:
  EcalDataFrame(); 

  virtual ~EcalDataFrame() {};    

  virtual const DetId& id() const=0;
    
  int size() const { return size_; }

  const EcalMGPASample& operator[](int i) const { return data_[i]; }
  const EcalMGPASample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  //    void setPresamples(int ps);
  void setSample(int i, const EcalMGPASample& sam) { data_[i]=sam; }

  static const int MAXSAMPLES = 10;

 protected:
  int size_;
  //    int ecalPresamples_;
  std::vector<EcalMGPASample> data_;    
};
  
#endif

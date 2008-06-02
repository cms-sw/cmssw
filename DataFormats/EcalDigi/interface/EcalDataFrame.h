#ifndef DIGIECAL_ECALDATAFRAME_H
#define DIGIECAL_ECALDATAFRAME_H

#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include <ostream>



/** \class EcalDataFrame
      
$Id: EcalDataFrame.h,v 1.1 2006/06/24 13:26:58 meridian Exp $
*/

class EcalDataFrame {
 public:
  EcalDataFrame(); 

  virtual ~EcalDataFrame() {};    

  virtual const DetId& id() const=0;
    
  int size() const { return size_; }

  const EcalMGPASample& operator[](const int& i) const { return data_[i]; }
  const EcalMGPASample& sample(const int& i) const { return data_[i]; }
    
  void setSize(const int& size);
  //    void setPresamples(int ps);
  void setSample(const int& i, const EcalMGPASample& sam) { data_[i]=sam; }

  static const int MAXSAMPLES = 10;

 protected:
  int size_;
  //    int ecalPresamples_;
  std::vector<EcalMGPASample> data_;    
};
  
#endif

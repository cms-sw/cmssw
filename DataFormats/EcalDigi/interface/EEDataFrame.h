#ifndef DIGIECAL_EEDATAFRAME_H
#define DIGIECAL_EEDATAFRAME_H

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include <vector>
#include <ostream>

/** \class EBDataFrame
      
$Id : $
*/

class EEDataFrame {
 public:
  typedef EEDetId key_type; ///< For the sorted collection

  EEDataFrame(); // for persistence
  explicit EEDataFrame(const EEDetId& id);
    
  const EEDetId& id() const { return id_; }
    
  int size() const { return size_; }

  const EcalMGPASample& operator[](int i) const { return data_[i]; }
  const EcalMGPASample& sample(int i) const { return data_[i]; }
    
  void setSize(int size);
  //    void setPresamples(int ps);
  void setSample(int i, const EcalMGPASample& sam) { data_[i]=sam; }

  static const int MAXSAMPLES = 10;

 private:
  EEDetId id_;
  int size_;
  //    int ecalPresamples_;
  std::vector<EcalMGPASample> data_;    
};
  

std::ostream& operator<<(std::ostream&, const EEDataFrame&);




#endif

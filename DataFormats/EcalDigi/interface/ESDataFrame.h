#ifndef DIGIECAL_ESDATAFRAME_H
#define DIGIECAL_ESDATAFRAME_H

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESSample.h"
#include <vector>
#include <ostream>

class ESDataFrame {

 public:

  typedef ESDetId key_type; ///< For the sorted collection

  ESDataFrame(); 
  explicit ESDataFrame(const ESDetId& id);
    
  const ESDetId& id() const { return id_; }
    
  int size() const { return size_; }

  const ESSample& operator[](const int& i) const { return data_[i]; }
  const ESSample& sample(const int& i) const { return data_[i]; }
    
  void setSize(const int& size);

  void setSample(const int& i, const ESSample& sam) { data_[i] = sam; }

  static const int MAXSAMPLES = 3;

 private:

  ESDetId id_;
  int size_;

  std::vector<ESSample> data_;    
};
  
std::ostream& operator<<(std::ostream&, const ESDataFrame&);

#endif
